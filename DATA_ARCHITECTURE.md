# Data Architecture Document

## 1. Objective

The pipeline converts messy technology-news records into an auditable analytical dataset containing company ARR observations over time and a searchable article representation. The assignment requires source lineage, enrichment, warehouse-style tables, CSV exports, and a downstream AI article dataset.

## 2. Architecture

```text
tech_news.csv -------------------+
                                  |
                                  v
                         [Raw / Source Layer]
                                  |
                                  v
                  [Clean + Standardize + Validate]
                   |        |        |        |
             Revenue     Dates    Category   Company
               |           |         |          |
               +-----------+---------+----------+
                                  |
                                  v
                         [Enriched Article Layer]
                                  |
                     +------------+------------+
                     |                         |
                     v                         v
                dim_company              fct_article
                                               |
                                               v
                                      fct_arr_observation
                                               |
                         +---------------------+----------------+
                         |                                      |
                         v                                      v
                ai_articles_enriched.csv                Vector artifacts
                                                        / similarity
```

## 3. Table grains and keys

### dim_company
One row per canonical company from `company_metadata.json`.

Primary key: `company_id`

### fct_article
One row per source article.

Primary/deterministic key: `article_key`

Business/source identifier: `article_id`

Important lineage columns:
- `source_file`
- `source_row_number`
- `raw_revenue`
- `revenue_parse_status`
- `revenue_parse_reason`
- `company_resolution_status`

### fct_arr_observation
One row per valid article ARR observation.

Key: deterministic `observation_id`

Foreign keys:
- `company_id`
- `article_id`
- `article_key`

This table intentionally excludes null, missing, N/A, and undisclosed revenue values.

## 4. Why ARR is a fact, not company master data

The assignment explicitly says every valid revenue value is interpreted as an ARR observation as of the article's publication date, while warning not to treat article ARR as company master data without lineage. Therefore, ARR lives in a fact table tied to the source article and date.

This preserves the distinction between:
- company attributes: `dim_company`
- article attributes: `fct_article`
- reported observations: `fct_arr_observation`

## 5. Revenue parsing

The parser supports:
- `$5.2B`
- `$5,200,000,000`
- `5.2 billion`
- `500M USD`
- EUR, GBP, JPY
- ranges such as `$10M - $20M`

Range midpoint is used. Currency conversion follows the fixed exercise rates.

The raw string is retained for audit.

## 6. Date normalization

The source includes ISO, US-style, EU-style, month-name, and ISO timestamp formats.

Rules:
- ISO `YYYY-MM-DD...` -> ISO date
- slash numeric -> `MM/DD/YYYY`
- hyphen numeric -> `DD-MM-YYYY` unless ISO
- other month-name formats are parsed by pandas

Derived fields:
- `year`
- `quarter`
- `month`

Invalid or missing dates remain null and are separately exported.

## 7. Category taxonomy

A controlled mapping converts equivalent source labels into:
- `AI_ML`
- `CLOUD`
- `DATA_ANALYTICS`
- `BIG_DATA`
- `FINTECH`
- `CYBERSECURITY`
- `SAAS`
- `SOFTWARE`
- `ENTERPRISE_SOFTWARE`
- `OTHER`

The mapping is explicit in source code rather than hidden in transformation logic.

## 8. Company identity resolution

Exact metadata names are preferred.

Known aliases are resolved to canonical metadata names. Unresolvable companies are not silently invented. They remain available in article data with `company_match_flag = False` and are exported to `unmatched_companies.csv`.

## 9. AI downstream dataset

The required output applies an OR condition:

`category_standardized = AI_ML OR industry = AI/ML`

Then filters:
- 2022-2024 inclusive
- valid ARR
- ARR > $50M

The output includes the assignment's requested fields plus the embedding and top-3 similar article IDs.

## 10. Semantic search

The interface supports:
- query-to-article similarity
- top-k similar article lookup
- hybrid filtering by date, ARR, and AI qualification

The supplied runnable implementation uses TF-IDF vectors as an offline-compatible equivalent. In an environment where model downloads are permitted, the same interface can use `sentence-transformers/all-MiniLM-L6-v2`.

## 11. Idempotency

The source `article_id` is stable, so:
- `article_key = SHA256(article_id)`
- `observation_id = SHA256(article_id)`

Reprocessing the same batch therefore does not create duplicate ARR facts. In a production warehouse, these keys would be enforced with unique constraints/merge logic.

## 12. Backfills

A production extension would:
1. land each source batch in a raw immutable layer;
2. add a batch/run identifier;
3. reprocess affected article IDs;
4. merge facts by deterministic observation key;
5. rebuild only impacted embedding records;
6. maintain parse-error/quarantine history.

If an article's source value changes, lineage should preserve both the prior and corrected source versions rather than silently overwriting historical evidence.

## 13. Schema evolution

Recommended production controls:
- explicit schema contract for source columns;
- nullable handling for new fields;
- validation before publishing gold outputs;
- versioned transformation logic;
- quarantine for unexpected data types/values;
- data-quality tests for unique IDs, required fields, valid dates, valid ARR, and referential integrity.

## 14. Scaling beyond local pandas

Pandas is appropriate for this exercise because the source is only 750 records and the assignment explicitly permits pandas. If source volume grows materially:
- replace pandas transformations with Spark DataFrames;
- move raw/source files to object storage;
- use partitioned columnar storage;
- use a warehouse/lakehouse for facts and dimensions;
- use a vector database or warehouse-native vector type for embeddings;
- schedule the batch through an orchestrator.

The logical model does not need to change merely because the execution engine changes.

## 15. Reliability / quality gates

Recommended production gates:
- all article IDs unique;
- no duplicate ARR observation IDs;
- valid ARR rows must have non-null article/company/date;
- no missing/undisclosed value may enter ARR fact;
- category values must belong to taxonomy;
- company foreign keys must resolve or be explicitly quarantined;
- AI output must satisfy the stated three filter conditions;
- embedding count must equal article count;
- top-similar lists must exclude the current article.

## 16. Current generated results

- Source rows: 750
- Valid ARR observations: 558
- AI downstream rows: 64
- Unmatched company rows: 25
- Invalid/missing date rows: 17

These figures are generated from the supplied source files and are included to make review reproducible.
