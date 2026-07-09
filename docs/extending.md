# Extending the Autonomous PM Engine

## Add a Data Source

1. Add a loader in `src/knowledge/document_loader.py`.
2. Add the file extension or connector type to the supported-source list.
3. Add metadata fields that preserve source, document type, and row/page information.

## Add an Agent

1. Create an agent module under `src/agents/`.
2. Add a LangGraph node in `src/orchestration/nodes.py`.
3. Register the node and edge in `src/orchestration/workflow.py`.
4. Add any new state fields to `src/orchestration/state.py`.

## Add an Output Format

1. Add a renderer in `src/output/`.
2. Add a template in `src/output/templates.py`.
3. Wire it into `PRDGenerator.generate_all()`.
