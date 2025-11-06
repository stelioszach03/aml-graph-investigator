PY=python
PIP=pip

.PHONY: install run synth ingest score test

install:
	$(PIP) install -r requirements.txt

run:
	uvicorn app.main:app --reload --port 8000

synth:
	$(PY) scripts/generate_synth.py --out data/raw/synth.csv --n 2000

ingest:
	$(PY) scripts/ingest_demo.py --input data/raw/synth.csv --graph data/interim/graph.pkl

score:
	$(PY) scripts/score_demo.py --graph data/interim/graph.pkl --out data/processed/scores.csv

test:
	$(PY) -m pytest -q

.PHONY: doctor docker-build docker-test docker-up docker-down up clean demo

doctor:
	bash scripts/dev_doctor.sh

docker-build:
	docker compose build --no-cache

docker-test:
	docker compose --profile test run --rm tests || (echo '\n❌ Tests failed'; exit 1)

docker-up:
	docker compose --profile dev up -d
	bash scripts/wait_for_http.sh http://localhost:8000/api/v1/health 90 || (echo 'API did not become healthy'; exit 1)
	@echo '\n✅ Stack is up.'
	@echo 'API: http://localhost:8000/api/v1/health'
	@echo 'UI : http://localhost:5173'
	@echo 'Neo4j Browser: http://localhost:7474 (neo4j/test)'

docker-down:
	docker compose down -v

up: doctor docker-build docker-test docker-up

clean:
	docker compose down -v || true
	docker image prune -f || true
	rm -rf ui/web/dist || true

demo:
	python scripts/generate_synth.py --n_accounts 2000 --fraud_rings 6 --seed 42
	python scripts/ingest_demo.py --path data/raw/synth_edges.csv --neo4j 0
	python -m app.ml.train_lgbm --features models/baseline/features.parquet --labels data/processed/labels.csv --out models/baseline
	python scripts/score_demo.py --topk 50

.PHONY: e2e neo4j-reset ui-build

e2e: docker-up
	bash scripts/smoke_e2e.sh

neo4j-reset:
	bash scripts/reset_neo4j.sh

ui-build:
	docker compose build ui

.PHONY: demo-run-docker
demo-run-docker:
	# Run pipeline inside api container (no host deps)
	docker compose exec -T api bash -lc '\
	set -euo pipefail; cd /work && \
	export API_BASE=http://api:8000 && \
	python scripts/generate_synth.py --n_accounts 3000 --fraud_rings 8 --seed 42 && \
	python scripts/ingest_demo.py --path data/raw/synth_edges.csv --neo4j 0 && \
	python -m app.ml.train_lgbm --features models/baseline/features.parquet --labels data/processed/labels.csv --out models/baseline && \
	python scripts/score_demo.py --topk 50 \
	'

.PHONY: set-token
set-token:
	@test -n "$$TOKEN" || (echo "Usage: make set-token TOKEN=xxxxx"; exit 2)
	@touch .env
	@grep -q '^API_AUTH_TOKEN=' .env && sed -i.bak -E "s/^API_AUTH_TOKEN=.*/API_AUTH_TOKEN=$$TOKEN/" .env || echo "API_AUTH_TOKEN=$$TOKEN" >> .env
	@echo "API_AUTH_TOKEN set in .env"

.PHONY: demo-open demo-protected demo-run

demo-open:
	# Clear API token → Open API
	sed -i.bak -E 's/^API_AUTH_TOKEN=.*/API_AUTH_TOKEN=/' .env
	@echo 'API_AUTH_TOKEN cleared (Open API).'
	make ui-build
	make up
	make demo-run
	@echo '\n💡 Open http://localhost:5173 and press "Refresh" in Scores.'

demo-protected:
	@test -n "$$TOKEN" || (echo "Usage: make demo-protected TOKEN=yourtoken"; exit 2)
	# Set backend token
	sed -i.bak -E 's/^API_AUTH_TOKEN=.*/API_AUTH_TOKEN='"$$TOKEN"'/' .env
	@echo 'API_AUTH_TOKEN set to '"$$TOKEN"
	# Pass default token to UI build so it auto-seeds localStorage
	docker compose build --build-arg VITE_DEFAULT_TOKEN=$$TOKEN ui
	make up
	make demo-run
	@echo '\n🔒 Protected demo ready. UI already seeded with token.'

demo-run:
	# Generate → ingest → train → score (makes Scores table show data)
	docker compose exec -T api python scripts/generate_synth.py --n_accounts 3000 --fraud_rings 8 --seed 42
	@TOKEN=$$(grep -E '^API_AUTH_TOKEN=' .env | sed -E 's/^API_AUTH_TOKEN=//'); \
	HDR_AUTH=""; \
	if [ -n "$$TOKEN" ]; then HDR_AUTH="-H Authorization: Bearer $$TOKEN"; fi; \
	curl -fsS -H "Content-Type: application/json" $$HDR_AUTH -d '{"path":"data/raw/synth_edges.csv","push_neo4j":false}' http://localhost:8000/api/v1/ingest; \
	curl -fsS -H "Content-Type: application/json" $$HDR_AUTH -d '{"labels_path":"data/processed/labels.csv"}' http://localhost:8000/api/v1/train; \
	curl -fsS -H "Content-Type: application/json" $$HDR_AUTH -d '{"topk":100}' http://localhost:8000/api/v1/score | head -c 200; echo
