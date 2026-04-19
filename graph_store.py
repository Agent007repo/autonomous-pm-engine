"""
src/knowledge/graph_store.py

Neo4j-backed entity linking layer.

Graph Schema
------------
Nodes:
  (:PainPoint  {id, text, doc_type, source, frequency})
  (:Feature    {id, name, description, category})
  (:Theme      {id, name})

Relationships:
  (:PainPoint)-[:MAPS_TO    {confidence: float}]->(:Feature)
  (:PainPoint)-[:BELONGS_TO               ]->(:Theme)
  (:Feature)  -[:PART_OF                  ]->(:Theme)
  (:PainPoint)-[:CO_OCCURS_WITH {count: int}]->(:PainPoint)

The entity extraction pipeline uses an LLM to:
  1. Identify pain points from each chunk.
  2. Match them to existing Feature nodes (or create new ones).
  3. Compute co-occurrence between pain points in the same chunk.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from loguru import logger
from neo4j import GraphDatabase, Driver

from src.config.settings import get_settings


@dataclass
class PainPoint:
    id: str
    text: str
    doc_type: str
    source: str
    frequency: int = 1


@dataclass
class Feature:
    id: str
    name: str
    description: str
    category: str


@dataclass
class EntityLinkingResult:
    pain_points: list[PainPoint]
    features: list[Feature]
    mappings: list[tuple[str, str, float]]   # (pain_point_id, feature_id, confidence)
    themes: list[str]


class GraphStore:
    """
    Manages all Neo4j operations for the PM Engine.
    Provides methods for entity extraction, graph population, and
    structured trend queries used by the Data Analyst Agent.
    """

    _EXTRACTION_PROMPT = """
You are a senior product analyst. Given the following customer feedback chunk,
extract all distinct pain points and map each to a product feature category.

Return ONLY a valid JSON object with this exact schema (no markdown, no explanation):
{
  "pain_points": [
    {
      "id": "pp_<short_slug>",
      "text": "<concise pain point description, max 20 words>",
      "feature": "<product feature this relates to, e.g. 'Search', 'Notifications', 'Onboarding'>",
      "feature_category": "<category: 'Core Feature' | 'UX' | 'Performance' | 'Integration' | 'Pricing'>",
      "theme": "<high-level theme, e.g. 'Discoverability', 'Speed', 'Reliability'>"
    }
  ]
}

CHUNK (doc_type={doc_type}, source={source}):
{text}
"""

    def __init__(self, driver: Driver | None = None) -> None:
        cfg = get_settings()
        self._cfg = cfg
        self._driver: Driver = driver or GraphDatabase.driver(
            cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password)
        )
        self._llm = ChatOpenAI(
            api_key=cfg.openai_api_key,
            model=cfg.openai_model,
            temperature=0.0,  # deterministic for extraction
        )
        self._init_schema()
        logger.info("GraphStore connected to Neo4j.")

    def close(self) -> None:
        self._driver.close()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        """Create uniqueness constraints and indexes on first run."""
        with self._driver.session() as session:
            queries = [
                "CREATE CONSTRAINT pp_id IF NOT EXISTS FOR (n:PainPoint) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT feat_id IF NOT EXISTS FOR (n:Feature) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT theme_id IF NOT EXISTS FOR (n:Theme) REQUIRE n.id IS UNIQUE",
                "CREATE INDEX pp_freq IF NOT EXISTS FOR (n:PainPoint) ON (n.frequency)",
            ]
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    logger.debug(f"Schema init note: {e}")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def extract_and_store(self, documents: list[Document]) -> int:
        """
        Run LLM-based entity extraction on each document chunk and
        write the resulting nodes + relationships to Neo4j.

        Returns the total number of pain points stored.
        """
        total = 0
        for doc in documents:
            try:
                result = self._extract_entities(doc)
                self._write_to_graph(result, doc)
                total += len(result.pain_points)
            except Exception as e:
                logger.warning(f"Entity extraction failed for chunk: {e}")
        logger.info(f"Entity extraction complete. Pain points stored: {total}")
        return total

    def _extract_entities(self, doc: Document) -> EntityLinkingResult:
        prompt = self._EXTRACTION_PROMPT.format(
            doc_type=doc.metadata.get("doc_type", "unknown"),
            source=doc.metadata.get("source", "unknown"),
            text=doc.page_content[:1500],  # Guard against token overflow
        )
        response = self._llm.invoke(prompt)
        raw = response.content.strip()

        # Strip markdown fences if the model adds them
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed: dict[str, Any] = json.loads(raw)
        pain_points: list[PainPoint] = []
        features: list[Feature] = []
        mappings: list[tuple[str, str, float]] = []
        themes: list[str] = []

        for item in parsed.get("pain_points", []):
            pp = PainPoint(
                id=item["id"],
                text=item["text"],
                doc_type=doc.metadata.get("doc_type", "unknown"),
                source=doc.metadata.get("source", "unknown"),
            )
            feat_id = f"feat_{item['feature'].lower().replace(' ', '_')}"
            feat = Feature(
                id=feat_id,
                name=item["feature"],
                description=item.get("text", ""),
                category=item.get("feature_category", "Core Feature"),
            )
            pain_points.append(pp)
            features.append(feat)
            mappings.append((pp.id, feat.id, 0.9))
            themes.append(item.get("theme", "General"))

        return EntityLinkingResult(
            pain_points=pain_points,
            features=features,
            mappings=mappings,
            themes=list(set(themes)),
        )

    def _write_to_graph(self, result: EntityLinkingResult, doc: Document) -> None:
        with self._driver.session() as session:
            # Upsert PainPoint nodes (MERGE on id, increment frequency)
            for pp in result.pain_points:
                session.run(
                    """
                    MERGE (p:PainPoint {id: $id})
                    ON CREATE SET p.text = $text, p.doc_type = $doc_type,
                                  p.source = $source, p.frequency = 1
                    ON MATCH  SET p.frequency = p.frequency + 1
                    """,
                    id=pp.id, text=pp.text, doc_type=pp.doc_type, source=pp.source,
                )

            # Upsert Feature nodes
            for feat in result.features:
                session.run(
                    """
                    MERGE (f:Feature {id: $id})
                    ON CREATE SET f.name = $name, f.description = $description,
                                  f.category = $category
                    """,
                    id=feat.id, name=feat.name,
                    description=feat.description, category=feat.category,
                )

            # Upsert MAPS_TO relationships
            for pp_id, feat_id, confidence in result.mappings:
                session.run(
                    """
                    MATCH (p:PainPoint {id: $pp_id}), (f:Feature {id: $feat_id})
                    MERGE (p)-[r:MAPS_TO]->(f)
                    ON CREATE SET r.confidence = $conf
                    ON MATCH  SET r.confidence = (r.confidence + $conf) / 2
                    """,
                    pp_id=pp_id, feat_id=feat_id, conf=confidence,
                )

            # Upsert Theme nodes and BELONGS_TO relationships
            for i, pp in enumerate(result.pain_points):
                if i < len(result.themes):
                    theme_name = result.themes[i]
                    theme_id = f"theme_{theme_name.lower().replace(' ', '_')}"
                    session.run(
                        """
                        MERGE (t:Theme {id: $theme_id})
                        ON CREATE SET t.name = $theme_name
                        WITH t
                        MATCH (p:PainPoint {id: $pp_id})
                        MERGE (p)-[:BELONGS_TO]->(t)
                        """,
                        theme_id=theme_id, theme_name=theme_name, pp_id=pp.id,
                    )

            # Co-occurrence: pain points from the same chunk are co-occurring
            pp_ids = [pp.id for pp in result.pain_points]
            for i, id_a in enumerate(pp_ids):
                for id_b in pp_ids[i + 1 :]:
                    session.run(
                        """
                        MATCH (a:PainPoint {id: $id_a}), (b:PainPoint {id: $id_b})
                        MERGE (a)-[r:CO_OCCURS_WITH]-(b)
                        ON CREATE SET r.count = 1
                        ON MATCH  SET r.count = r.count + 1
                        """,
                        id_a=id_a, id_b=id_b,
                    )

    # ── Query API (used by Data Analyst Agent) ───────────────────────────────

    def get_top_pain_points(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most frequently mentioned pain points."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (p:PainPoint)
                RETURN p.id AS id, p.text AS text, p.frequency AS frequency,
                       p.doc_type AS doc_type
                ORDER BY p.frequency DESC
                LIMIT $limit
                """,
                limit=limit,
            )
            return [dict(r) for r in result]

    def get_feature_impact_summary(self) -> list[dict[str, Any]]:
        """
        For each Feature, return how many distinct pain points map to it
        and the total pain-point frequency (a proxy for impact).
        """
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (p:PainPoint)-[:MAPS_TO]->(f:Feature)
                RETURN f.id AS feature_id,
                       f.name AS feature_name,
                       f.category AS category,
                       COUNT(DISTINCT p) AS pain_point_count,
                       SUM(p.frequency) AS total_frequency
                ORDER BY total_frequency DESC
                """
            )
            return [dict(r) for r in result]

    def get_theme_summary(self) -> list[dict[str, Any]]:
        """Return themes ranked by the count of pain points belonging to them."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (p:PainPoint)-[:BELONGS_TO]->(t:Theme)
                RETURN t.name AS theme,
                       COUNT(p) AS pain_point_count,
                       SUM(p.frequency) AS total_mentions
                ORDER BY total_mentions DESC
                """
            )
            return [dict(r) for r in result]

    def get_co_occurring_pain_points(
        self, min_count: int = 2
    ) -> list[dict[str, Any]]:
        """Return pairs of pain points that co-occur frequently."""
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (a:PainPoint)-[r:CO_OCCURS_WITH]-(b:PainPoint)
                WHERE r.count >= $min_count AND id(a) < id(b)
                RETURN a.text AS pain_a, b.text AS pain_b, r.count AS co_count
                ORDER BY r.count DESC
                LIMIT 20
                """,
                min_count=min_count,
            )
            return [dict(r) for r in result]
