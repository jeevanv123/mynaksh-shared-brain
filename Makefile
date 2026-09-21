.PHONY: install run run-offline test test-integration neo4j-up neo4j-down demo

PY ?= .venv/bin/python

install:
	python3 -m venv .venv && $(PY) -m pip install -q -r requirements.txt

run:            ## real LLM + Neo4j (reads .env)
	$(PY) -m uvicorn app.main:app --reload --port 8000

run-offline:    ## no key, no Docker: fake LLM + in-memory graph
	LLM_PROVIDER=fake GRAPH_STORE=memory $(PY) -m uvicorn app.main:app --reload --port 8000

test:
	$(PY) -m pytest

test-integration:  ## needs Neo4j on localhost:7687 (make neo4j-up)
	RUN_NEO4J_TESTS=1 $(PY) -m pytest tests/test_neo4j_integration.py -v

neo4j-up:
	docker compose up -d neo4j

neo4j-down:
	docker compose down

demo:           ## scripted walkthrough against a running server on :8000
	bash scripts/demo.sh
