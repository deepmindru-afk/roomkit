.PHONY: audit install lint format typecheck security test test-changed api-snapshot coverage stress llms-full check-models update-llamacpp all clean docs docs-imports docs-serve llms-sync llms-check conformance-sync deploy deploy-status release

install:
	uv sync --extra dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

typecheck:
	uv run ty check src/roomkit/
	uv run pytest -m typing -q tests/test_typing_subclasses.py tests/test_typing_lazy_exports.py

security:
	uv run bandit -r src/ -c pyproject.toml

# Same two passes CI runs, so a new advisory can be seen before pushing:
# the core gates, the extras only report.
#
# --python pins the interpreter pip-audit resolves against: 3.12, this project's
# floor and what CI installs. Without it uvx may pick an older one, and the pass
# then dies on "no matching distribution" for any dependency requiring >=3.12 —
# a resolution failure that reads exactly like a clean run in the extras pass,
# since that one only reports.
#
# The requirement files go to a directory of their own under $TMPDIR, removed
# on exit, so two runs never share one and nothing is left in /tmp.
audit:
	tmp=$$(mktemp -d) && trap 'rm -rf "$$tmp"' EXIT && \
	uv export --frozen --no-dev --no-emit-project --format requirements-txt -o "$$tmp/core.txt" && \
	uvx --python 3.12 pip-audit --requirement "$$tmp/core.txt" && \
	{ uv export --frozen --no-dev --no-emit-project --extra all --no-hashes --format requirements-txt -o "$$tmp/all.txt" && \
	  uvx --python 3.12 pip-audit --requirement "$$tmp/all.txt" || \
	  echo "audit: the extras pass failed or found advisories (reported, not gating)"; }

# One worker per core (pytest-xdist); an idle worker takes tests queued for a
# busy one, since a few tests wait seconds in real time (20% sooner than an
# even split). Not in addopts: a bare pytest, as the smoke job runs, lacks it.
PARALLEL = -n auto --dist worksteal

# The whole suite.
test:
	uv run pytest $(PARALLEL)

# Only the tests whose code changed since the last run (pytest-testmon; the
# first run records, and runs everything). A dependency testmon cannot see —
# a data file, a module installed into sys.modules — needs `make test`, which
# `make all` and CI run.
test-changed:
	uv run pytest $(PARALLEL) --testmon --testmon-forceselect

# Rewrite tests/public_api_signatures.json and tests/public_api_names.json
# from the code. A public signature change that can break a caller, or a name
# dropped from a package's __all__, fails test_public_api.py until this runs:
# the commit that breaks on purpose runs it and carries its BREAKING entry.
# It runs in an environment of its own installed with --extra all (made on
# first use), so a name only an extra of that set resolves, roomkit.webrtc's,
# is pinned too; extras outside it (livekit, docs, pynacl, webtransport, the
# local model runtimes neutts, qwen-*, pocket-tts, phonon, vui...) are not.
# The dev environment's guard leaves out what its missing extras resolve.
API_SNAPSHOT_ENV ?= .venv-api-snapshot
api-snapshot:
	UV_PROJECT_ENVIRONMENT=$(API_SNAPSHOT_ENV) uv run --frozen --extra all python -m tests.api_signatures

coverage:
	uv run pytest $(PARALLEL) --cov=roomkit --cov-report=term-missing

stress:
	uv run pytest -m stress -v

llms-full:
	uv run python scripts/build_llms_full.py

# Compare providers/*/models.py against a live upstream mirror. Nothing in the
# test suite can catch a vendor shipping a new flagship — a stale catalog is
# self-consistent. `make release` runs this; run it by hand any time.
check-models:
	uv run python scripts/check_models.py

# Pin the newest llama.cpp build the llamacpp provider downloads (archives and
# SHA-256 from the GitHub release). Pass a tag to pin another: BUILD=b11160.
update-llamacpp:
	uv run python scripts/update_llamacpp_build.py $(BUILD)
	uv run ruff format src/roomkit/providers/llamacpp/_builds.py

all: lint typecheck security test

clean:
	rm -rf dist/ build/ .pytest_cache/ .ruff_cache/ htmlcov/ .coverage .testmondata*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true

# --- Documentation ---

DOCS_DIR ?= ../roomkit-docs

docs: llms-check docs-imports
	uv run --extra docs mkdocs build --strict -f $(DOCS_DIR)/mkdocs.yml

# The docs site's pages import roomkit names from public paths only, as
# tests/test_public_imports.py checks roomkit's own examples and documents.
docs-imports:
	uv run python -m tests.public_imports $(DOCS_DIR)/docs

docs-serve:
	uv run --extra docs mkdocs serve -f $(DOCS_DIR)/mkdocs.yml

# llms.txt and llms-full.txt are written here (llms-full.txt from docs/c7); the
# docs site serves copies of them. Never edit the copies in roomkit-docs.
llms-sync: llms-full
	cp llms.txt llms-full.txt $(DOCS_DIR)/docs/

llms-check:
	@cmp -s llms.txt $(DOCS_DIR)/docs/llms.txt && cmp -s llms-full.txt $(DOCS_DIR)/docs/llms-full.txt \
		|| { echo "$(DOCS_DIR)/docs/llms*.txt differ from roomkit's: run 'make llms-sync' and commit the copies"; exit 1; }

# The shared conformance suite (roomkit-specs conformance/): its scenarios at
# one roomkit-specs commit, copied into tests/conformance/scenarios with that
# commit recorded in tests/conformance/SPECS_COMMIT, so CI needs no sibling
# checkout. The copy is never edited here: change the scenarios in
# roomkit-specs, commit them there, and sync again. SPECS_REF names the commit.
SPECS_DIR ?= ../roomkit-specs
SPECS_REF ?= HEAD

conformance-sync:
	@commit=$$(git -C $(SPECS_DIR) rev-parse --verify "$(SPECS_REF)^{commit}") && \
	rm -rf tests/conformance/scenarios && \
	git -C $(SPECS_DIR) archive --format=tar "$$commit" conformance/scenarios \
		| tar -x --strip-components=1 -C tests/conformance && \
	echo "$$commit" > tests/conformance/SPECS_COMMIT && \
	echo "tests/conformance synced from roomkit-specs $$commit"

# --- Website deployment ---

deploy:
	./deploy.sh full

deploy-status:
	./deploy.sh status

# --- Release (bump, test, build, publish, tag, push) ---

release:
ifndef VERSION
	$(error VERSION is required. Usage: make release VERSION=0.4.1)
endif
	./scripts/release.sh $(VERSION)
