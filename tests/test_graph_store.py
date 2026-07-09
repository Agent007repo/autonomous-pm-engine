"""Smoke tests for graph-store imports.

GraphStore itself requires a running Neo4j instance and OpenAI credentials, so
unit tests should mock those dependencies before instantiation. This smoke test
keeps the documented test path importable without requiring external services.
"""


def test_graph_store_module_importable():
    import src.knowledge.graph_store as graph_store

    assert hasattr(graph_store, "GraphStore")
