# Agent Guidance

## Project Shape

- This is a Python/pandas data-engineering pipeline, not a frontend application.
- The main implementation is in [`src/pipeline.py`](src/pipeline.py); focused unit tests are in [`tests/test_pipeline.py`](tests/test_pipeline.py).
- Read [`README.md`](README.md) for the runnable workflow and [`DATA_ARCHITECTURE.md`](DATA_ARCHITECTURE.md) for table grains, keys, lineage, and quality rules.

## Commands

- Install dependencies: `python -m pip install -r requirements.txt`
- Run tests: `pytest -q`
- Run the pipeline: `python src/pipeline.py --news tech_news.csv --metadata company_metadata.json --output-dir outputs`

Run the focused tests after changes to parsing, date normalization, category mapping, company resolution, table construction, or search behavior. Preserve deterministic outputs and avoid modifying generated files unless the task explicitly targets them.

## Implementation Conventions

- Keep transformations modular and deterministic; prefer the existing functions in `src/pipeline.py` over new parallel abstractions.
- Preserve source lineage, raw revenue, parse statuses, deterministic IDs, and explicit unmatched-company handling.
- Treat ARR as an article-dated observation, not a company master attribute.
- Keep missing or undisclosed revenue out of `fct_arr_observation`.
- Use the documented fixed currency rates, date assumptions, category taxonomy, and AI-output filters.
- Keep the offline TF-IDF embedding path compatible with the existing similarity-search interfaces.

## Visual Theme

- This repository currently has no UI or dark-mode implementation.
- If a future visualization, report, or interface is added, use light mode as the default and do not introduce a dark theme unless the user explicitly requests one.
- Keep visual additions consistent with the data product and validate them separately from the pipeline tests.

## Change Validation

- Prefer small, focused edits and run `pytest -q` after implementation changes.
- For changes affecting generated outputs, verify row counts, deterministic keys, null handling, and the quality gates documented in [`DATA_ARCHITECTURE.md`](DATA_ARCHITECTURE.md).
