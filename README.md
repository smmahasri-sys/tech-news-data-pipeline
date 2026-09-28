# YipitData Data Engineering Assignment

## Approach

This submission uses **Python + pandas** rather than PySpark.

The assignment explicitly allows either implementation. The input is 750 synthetic articles, which is small enough for an in-memory pandas pipeline while still allowing the code to be structured as reusable functions that can later be moved to Spark if scale requires it. The implementation is intentionally modular: parsing, date normalization, category standardization, company resolution, enrichment, modeling, embeddings, and exports are separate steps.

## What the pipeline does

1. Reads `tech_news.csv` and `company_metadata.json`.
2. Preserves source lineage using `source_file` and `source_row_number`.
3. Parses messy revenue values into integer `arr_usd` values.
4. Converts EUR/GBP/JPY using the assignment's fixed rates.
5. Uses the midpoint for revenue ranges.
6. Keeps missing/undisclosed revenue out of the ARR fact table.
7. Normalizes dates and derives year/quarter/month.
8. Standardizes categories into a controlled taxonomy.
9. Resolves company aliases and flags unmatched companies.
10. Enriches articles from the supplied company metadata.
11. Builds warehouse-style company, article, and ARR-observation tables.
12. Generates Transformer embeddings and top-3 similar article IDs.
13. Produces the required `ai_articles_enriched.csv`.
14. Optionally creates a DuckDB database when DuckDB is installed.

## Data model

### `dim_company`
**Grain:** one row per canonical company in the supplied metadata.

### `fct_article`
**Grain:** one row per source article.

Contains source fields, standardized fields, enrichment, parse status, raw revenue, normalized ARR, embedding, and similarity lineage.

### `fct_arr_observation`
**Grain:** one valid ARR observation per source article.

`observation_id` is deterministic from `article_id`, making reruns idempotent.

Relationship:

`dim_company.company_id -> fct_article.company_id -> fct_arr_observation.company_id`

`fct_arr_observation.article_id -> fct_article.article_id`

## Revenue assumptions

The assignment specifies:
- EUR -> USD: multiply by 1.1
- GBP -> USD: multiply by 1.27
- JPY -> USD: divide by 150
- ranges use midpoint
- missing / N/A / Not disclosed are invalid ARR

The original source value is retained as `raw_revenue` for audit.

## Date assumption

ISO dates are parsed as ISO. Slash-separated numeric dates are interpreted as US `MM/DD/YYYY`; hyphen-separated numeric dates such as `23-08-2023` are interpreted as EU `DD-MM-YYYY`. This is documented because ambiguous dates cannot be resolved from the raw string alone.

## Company resolution

Aliases such as AWS, Open AI, Nvidia, Mongo DB, and similar source variants are mapped to canonical metadata companies. `Microsoft Azure` and `Azure` are treated as Microsoft for this synthetic exercise because the supplied metadata contains Microsoft but not Azure as a separate company.

Companies for which no supplied metadata identity exists are retained in article outputs but flagged in `unmatched_companies.csv`; metadata fields remain null rather than being invented.

## AI output rule

`ai_articles_enriched.csv` contains records where:

- standardized article category is `AI_ML` **OR** metadata industry is `AI/ML`;
- year is 2022, 2023, or 2024;
- `arr_usd > 50,000,000`;
- ARR is valid.

## Embeddings / semantic search

The production code exposes the requested `find_similar_articles` and `hybrid_search` interfaces.

Embeddings are generated with `sentence-transformers/all-MiniLM-L6-v2`, loaded directly through the Hugging Face Transformers library. The pipeline mean-pools token embeddings, normalizes the resulting 384-dimensional vectors, and uses their dot product as cosine similarity. The model must be available locally or downloadable from Hugging Face on the first run.

`article_embeddings.npy` stores the reusable matrix, while the modeled CSV stores each embedding as JSON.

## Idempotency / backfills

- Article records are keyed by deterministic `article_key` derived from `article_id`.
- ARR observations use deterministic `observation_id` derived from `article_id`.
- ARR observations are de-duplicated by `observation_id`.
- The pipeline overwrites/rebuilds output tables deterministically, so rerunning the same source batch does not create duplicate modeled records.
- In a production incremental system, the same keys can be enforced as warehouse primary/natural keys and new source batches can be merged/upserted.

## Run

```bash
python src/pipeline.py --news tech_news.csv --metadata company_metadata.json --output-dir outputs
```

## Tests

```bash
pytest -q
```

## Example analytical queries

Latest ARR per company can be derived from `fct_arr_observation` by ordering by `published_date_normalized` and selecting the latest observation per company.

Quarterly ARR observations can be derived using `company_id`, `year`, and `quarter`.

Example pandas:

```python
latest = (
    fct_arr.sort_values("published_date_normalized")
           .groupby("company_id", as_index=False)
           .tail(1)
)

quarterly = (
    fct_arr.groupby(["company_id", "year", "quarter"], as_index=False)
           .agg(arr_usd=("arr_usd", "max"))
)
```

## Submission contents

- `src/pipeline.py`
- `tests/test_pipeline.py`
- `README.md`
- `DATA_ARCHITECTURE.md`
- `requirements.txt`
- `outputs/ai_articles_enriched.csv`
- `outputs/dim_company.csv`
- `outputs/fct_article.csv`
- `outputs/fct_arr_observation.csv`
- `outputs/unmatched_companies.csv`
- `outputs/invalid_dates.csv`
- `outputs/article_embeddings.npy`
- `outputs/article_embeddings.json`
- optional `outputs/yipitdata.duckdb` if DuckDB is installed

## Generated dataset statistics

- Source articles: 750
- Canonical metadata companies: 21
- Valid ARR observations: 558
- Unmatched company records: 25
- Invalid/missing dates: 17
- AI downstream records: 64
