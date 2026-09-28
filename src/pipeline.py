
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # optional but preferred for semantic search
    torch = None
    AutoModel = None
    AutoTokenizer = None

try:
    import duckdb
except ImportError:  # optional local analytical store
    duckdb = None


CURRENCY_RATES = {"USD": 1.0, "EUR": 1.10, "GBP": 1.27, "JPY": 1 / 150}
MISSING_REVENUE = {"", "N/A", "NA", "NULL", "NONE", "NOT DISCLOSED", "NOT DISCLOSED."}

CATEGORY_MAP = {
    "AI & ML": "AI_ML",
    "AI/ML": "AI_ML",
    "Artificial Intelligence": "AI_ML",
    "Machine Learning": "AI_ML",
    "Cloud": "CLOUD",
    "Cloud Computing": "CLOUD",
    "Cloud Services": "CLOUD",
    "Data Analytics": "DATA_ANALYTICS",
    "Analytics": "DATA_ANALYTICS",
    "Big Data": "BIG_DATA",
    "Financial Technology": "FINTECH",
    "FinTech": "FINTECH",
    "Finance": "FINTECH",
    "Cybersecurity": "CYBERSECURITY",
    "Security": "CYBERSECURITY",
    "InfoSec": "CYBERSECURITY",
    "SaaS": "SAAS",
    "Software": "SOFTWARE",
    "Enterprise Software": "ENTERPRISE_SOFTWARE",
}

# Conservative aliases: only variants that can be resolved to a company
# identity represented in the supplied metadata are mapped.
COMPANY_ALIASES = {
    "AWS": "Amazon Web Services",
    "Amazon Web Services (AWS)": "Amazon Web Services",
    "DeepMind": "Google DeepMind",
    "Google Deepmind": "Google DeepMind",
    "Meta AI Research": "Meta AI",
    "Facebook AI Research": "Meta AI",
    "Nvidia": "NVIDIA",
    "NVIDIA Corporation": "NVIDIA",
    "Open AI": "OpenAI",
    "OpenAI Inc.": "OpenAI",
    "Databricks Inc.": "Databricks",
    "Snowflake Inc.": "Snowflake",
    "CloudFlare": "Cloudflare",
    "Data Robot": "DataRobot",
    "Mongo DB": "MongoDB",
    "Palantir Technologies": "Palantir",
    "Stripe Inc.": "Stripe",
    "The Boring Company / SpaceX": "SpaceX",
    # Product/business-unit naming used by the synthetic source.
    "Microsoft Azure": "Microsoft",
    "Azure": "Microsoft",
}


def parse_revenue(value: object) -> Tuple[Optional[int], str, str]:
    """Return (arr_usd, parse_status, parse_reason).

    Missing/undisclosed values are never modeled as ARR.
    Ranges use the midpoint. Currency conversion uses assignment rates.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, "missing", "null revenue"

    raw = str(value).strip()
    if raw.upper() in MISSING_REVENUE:
        return None, "missing", "missing_or_undisclosed"

    text = raw.replace(",", "").strip()
    upper = text.upper()

    # Determine currency from explicit code/symbol.
    if "EUR" in upper or "€" in text:
        currency = "EUR"
    elif "GBP" in upper or "£" in text:
        currency = "GBP"
    elif "JPY" in upper or "¥" in text:
        currency = "JPY"
    else:
        currency = "USD"

    # Remove currency markers; retain numbers, decimal points, range hyphens.
    cleaned = re.sub(r"(?i)USD|EUR|GBP|JPY", "", text)
    cleaned = cleaned.replace("$", "").replace("€", "").replace("£", "").replace("¥", "")
    cleaned = cleaned.strip()

    # Parse one or two numeric values. Scientific notation is not expected
    # in the supplied exercise, but ordinary decimals are supported.
    nums = re.findall(r"\d+(?:\.\d+)?", cleaned)
    if not nums:
        return None, "invalid", "no_numeric_amount"

    numbers = [float(x) for x in nums[:2]]
    amount = sum(numbers) / len(numbers) if len(numbers) == 2 and re.search(r"\s-\s|-\s|\s-", cleaned) else numbers[0]

    lower = cleaned.lower().strip()
    # Check long-form units first, then compact suffixes attached to a number.
    if "trillion" in lower or lower.endswith("t"):
        multiplier = 1_000_000_000_000
    elif "billion" in lower or lower.endswith("b"):
        multiplier = 1_000_000_000
    elif "million" in lower or lower.endswith("m"):
        multiplier = 1_000_000
    elif "thousand" in lower or lower.endswith("k"):
        multiplier = 1_000
    else:
        multiplier = 1

    usd = amount * multiplier * CURRENCY_RATES[currency]
    return int(round(usd)), "parsed", "range_midpoint" if len(numbers) == 2 else "single_value"


def normalize_date(value: object) -> Tuple[pd.Timestamp, str]:
    """Normalize source dates.

    Assumption for ambiguous numeric dates:
      - slash format is interpreted as US MM/DD/YYYY;
      - hyphen numeric format is interpreted as EU DD-MM-YYYY when it
        is not ISO YYYY-MM-DD.
    """
    raw = "" if value is None else str(value).strip()
    if not raw:
        return pd.NaT, "missing"

    # ISO date / ISO timestamp.
    if re.match(r"^\d{4}-\d{2}-\d{2}", raw):
        ts = pd.to_datetime(raw[:10], errors="coerce")
    # Explicit slash format: assignment contains US-style examples.
    elif re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", raw):
        ts = pd.to_datetime(raw, format="%m/%d/%Y", errors="coerce")
    # Numeric hyphen date: use DD-MM-YYYY unless first token is 4 digits.
    elif re.match(r"^\d{1,2}-\d{1,2}-\d{4}$", raw):
        ts = pd.to_datetime(raw, format="%d-%m-%Y", errors="coerce")
    else:
        ts = pd.to_datetime(raw, errors="coerce", dayfirst=False)

    return ts, "parsed" if pd.notna(ts) else "invalid"


def standardize_category(value: object) -> Tuple[str, str]:
    raw = "" if value is None else str(value).strip()
    if raw in CATEGORY_MAP:
        return CATEGORY_MAP[raw], "mapped"
    return "OTHER", "unmapped"


def resolve_company(name: object, metadata: dict) -> Tuple[Optional[str], str]:
    raw = "" if name is None else str(name).strip()
    if raw in metadata:
        return raw, "exact"
    if raw in COMPANY_ALIASES and COMPANY_ALIASES[raw] in metadata:
        return COMPANY_ALIASES[raw], "alias"
    return None, "unmatched"


def company_size(employee_count: object) -> str:
    if pd.isna(employee_count):
        return "UNKNOWN"
    n = int(employee_count)
    if n < 10_000:
        return "SMALL"
    if n <= 30_000:
        return "MEDIUM"
    return "LARGE"


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


class TransformerEmbeddingBackend:
    """Encode text with all-MiniLM-L6-v2 without sentence-transformers."""

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts: list[str], convert_to_numpy: bool = True,
               normalize_embeddings: bool = True) -> np.ndarray:
        tokens = self.tokenizer(texts, padding=True, truncation=True,
                                return_tensors="pt")
        with torch.no_grad():
            hidden = self.model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        result = embeddings.cpu().numpy()
        return result if convert_to_numpy else embeddings


def clean_and_enrich_articles(news_path: str | Path, metadata_path: str | Path):
    news = pd.read_csv(news_path, dtype=str, keep_default_na=False)
    news.insert(1, "source_row_number", np.arange(2, len(news) + 2))
    metadata = json.loads(Path(metadata_path).read_text())

    revenue_result = news["revenue"].apply(parse_revenue)
    news["arr_usd"] = revenue_result.map(lambda x: x[0]).astype("Int64")
    news["revenue_parse_status"] = revenue_result.map(lambda x: x[1])
    news["revenue_parse_reason"] = revenue_result.map(lambda x: x[2])

    date_result = news["published_date"].apply(normalize_date)
    news["published_date_normalized"] = pd.to_datetime(date_result.map(lambda x: x[0]), errors="coerce")
    news["date_parse_status"] = date_result.map(lambda x: x[1])
    news["year"] = news["published_date_normalized"].dt.year.astype("Int64")
    news["quarter"] = news["published_date_normalized"].dt.quarter.astype("Int64").map(
        lambda x: f"Q{int(x)}" if pd.notna(x) else pd.NA
    )
    news["month"] = news["published_date_normalized"].dt.month.astype("Int64")

    cat = news["category"].apply(standardize_category)
    news["category_standardized"] = cat.map(lambda x: x[0])
    news["category_parse_status"] = cat.map(lambda x: x[1])

    resolved = news["company_name"].apply(lambda x: resolve_company(x, metadata))
    news["company_id"] = resolved.map(lambda x: x[0])
    news["company_resolution_status"] = resolved.map(lambda x: x[1])
    news["company_match_flag"] = news["company_id"].notna()

    def enrich(row):
        if pd.isna(row["company_id"]) or not row["company_id"]:
            return pd.Series({
                "industry": pd.NA, "founded_year": pd.NA, "headquarters": pd.NA,
                "employee_count": pd.NA, "is_public": pd.NA, "stock_ticker": pd.NA,
                "company_age": pd.NA, "company_size_category": "UNKNOWN",
            })
        company_id = row["company_id"]
        if company_id not in metadata:
            return pd.Series({
                "industry": pd.NA, "founded_year": pd.NA, "headquarters": pd.NA,
                "employee_count": pd.NA, "is_public": pd.NA, "stock_ticker": pd.NA,
                "company_age": pd.NA, "company_size_category": "UNKNOWN",
            })
        m = metadata[company_id]
        age = int(row["year"]) - int(m["founded_year"]) if pd.notna(row["year"]) else pd.NA
        return pd.Series({
            "industry": m["industry"],
            "founded_year": m["founded_year"],
            "headquarters": m["headquarters"],
            "employee_count": m["employee_count"],
            "is_public": m["is_public"],
            "stock_ticker": m["stock_ticker"],
            "company_age": age,
            "company_size_category": company_size(m["employee_count"]),
        })

    enriched = pd.concat([news, news.apply(enrich, axis=1)], axis=1)

    # Deterministic record key makes reruns idempotent.
    enriched["article_key"] = enriched["article_id"].map(lambda x: stable_id("ART", str(x)))
    enriched["source_file"] = Path(news_path).name

    # Source lineage is intentionally retained in the modeled article table.
    enriched["raw_revenue"] = enriched["revenue"]

    return enriched, metadata


def build_embeddings(articles: pd.DataFrame, model_path: str | Path):
    """Build embeddings for semantic article search.

    Preferred backend: sentence-transformers/all-MiniLM-L6-v2.
    Uses the Transformers model directly to avoid sentence-transformers import
    issues in environments where its optional integrations are incompatible.
    """
    text = (articles["title"].fillna("") + " " + articles["summary"].fillna("")).str.strip()
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if AutoModel is not None and AutoTokenizer is not None and torch is not None:
        model = TransformerEmbeddingBackend("sentence-transformers/all-MiniLM-L6-v2")
        dense = model.encode(text.tolist(), convert_to_numpy=True, normalize_embeddings=True)
        dense = np.asarray(dense, dtype=np.float32)
        metadata = {"type": "sentence-transformers", "model": "sentence-transformers/all-MiniLM-L6-v2"}
        backend = model
    else:
        raise RuntimeError(
            "Transformers backend is unavailable. Install torch and transformers "
            "in the same Python environment as the pipeline."
        )

    np.save(model_path.with_suffix(".npy"), dense)
    with model_path.with_suffix(".json").open("w") as f:
        json.dump(metadata, f)

    articles = articles.copy()
    articles["embedding"] = [json.dumps(v.tolist(), separators=(",", ":")) for v in dense]

    sim = dense @ dense.T
    top_ids = []
    ids = articles["article_id"].tolist()
    for i in range(len(ids)):
        scores = sim[i].copy()
        scores[i] = -np.inf
        k = min(3, len(ids) - 1)
        idx = np.argsort(-scores)[:k]
        top_ids.append(json.dumps([ids[j] for j in idx]))
    articles["top_similar_articles"] = top_ids
    return articles, dense, backend


def build_tables(enriched: pd.DataFrame, metadata: dict):
    dim_company = pd.DataFrame([
        {
            "company_id": k,
            "company_name": k,
            "industry": v["industry"],
            "founded_year": v["founded_year"],
            "headquarters": v["headquarters"],
            "employee_count": v["employee_count"],
            "is_public": v["is_public"],
            "stock_ticker": v["stock_ticker"],
            "company_size_category": company_size(v["employee_count"]),
        }
        for k, v in metadata.items()
    ])

    article_cols = [
        "article_key", "article_id", "source_row_number", "source_file",
        "title", "company_id", "company_name", "published_date_normalized",
        "year", "quarter", "month", "category", "category_standardized",
        "summary", "url", "author", "word_count", "raw_revenue", "arr_usd",
        "revenue_parse_status", "revenue_parse_reason", "date_parse_status",
        "company_resolution_status", "company_match_flag", "industry",
        "founded_year", "headquarters", "employee_count", "is_public",
        "stock_ticker", "company_age", "company_size_category",
        "embedding", "top_similar_articles",
    ]
    fct_article = enriched[article_cols].copy()

    valid = enriched[enriched["arr_usd"].notna()].copy()
    valid["observation_id"] = valid["article_id"].map(lambda x: stable_id("ARR", str(x)))
    fct_arr = valid[[
        "observation_id", "article_id", "article_key", "company_id", "company_name",
        "published_date_normalized", "year", "quarter", "month", "arr_usd",
        "raw_revenue", "revenue_parse_status", "source_file", "source_row_number",
    ]].copy()

    # Natural/deterministic key protects against duplicate records on rerun.
    fct_arr = fct_arr.drop_duplicates(subset=["observation_id"]).sort_values(
        ["company_id", "published_date_normalized", "article_id"]
    )
    return dim_company, fct_article, fct_arr


def _encode_query(embedding_backend, query_text: str) -> np.ndarray:
    if hasattr(embedding_backend, "encode"):
        q = embedding_backend.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(q, dtype=np.float32).reshape(1, -1)
    q = embedding_backend.transform([query_text]).toarray().astype(np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm:
        q = q / q_norm
    return q


def find_similar_articles(query_text: str, articles: pd.DataFrame, matrix: np.ndarray, embedding_backend, top_k: int = 5):
    q = _encode_query(embedding_backend, query_text)
    scores = (matrix @ q.T).ravel()
    idx = np.argsort(-scores)[:top_k]
    return [{"article_id": articles.iloc[i]["article_id"], "similarity": float(scores[i])} for i in idx]


def hybrid_search(query_text, articles, matrix, embedding_backend, top_k=5, year_start=2022,
                  year_end=2024, arr_min=50_000_000, ai_only=True):
    candidates = articles.copy()
    candidates = candidates[candidates["year"].between(year_start, year_end, inclusive="both")]
    candidates = candidates[candidates["arr_usd"].notna() & (candidates["arr_usd"] > arr_min)]
    if ai_only:
        candidates = candidates[
            (candidates["category_standardized"] == "AI_ML") |
            (candidates["industry"].eq("AI/ML"))
        ]
    if candidates.empty:
        return candidates[["article_id"]].assign(similarity=pd.Series(dtype=float))

    q = _encode_query(embedding_backend, query_text)
    positions = candidates.index.to_numpy()
    scores = (matrix[positions] @ q.T).ravel()
    result = candidates[["article_id", "title", "company_name", "year", "arr_usd"]].copy()
    result["similarity"] = scores
    return result.sort_values("similarity", ascending=False).head(top_k)


def write_duckdb(tables: dict[str, pd.DataFrame], output_path: str | Path):
    if duckdb is None:
        return False
    con = duckdb.connect(str(output_path))
    try:
        for name, table in tables.items():
            con.register(f"tmp_{name}", table)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp_{name}")
            con.unregister(f"tmp_{name}")
    finally:
        con.close()
    return True


def run_pipeline(news_path: str, metadata_path: str, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    enriched, metadata = clean_and_enrich_articles(news_path, metadata_path)

    enriched, matrix, embedding_backend = build_embeddings(enriched, out / "article_embeddings")
    dim_company, fct_article, fct_arr = build_tables(enriched, metadata)

    # Canonical output names.
    dim_company.to_csv(out / "dim_company.csv", index=False)
    fct_article.to_csv(out / "fct_article.csv", index=False)
    fct_arr.to_csv(out / "fct_arr_observation.csv", index=False)

    ai = enriched[
        (
            enriched["category_standardized"].eq("AI_ML") |
            enriched["industry"].eq("AI/ML")
        )
        & enriched["year"].between(2022, 2023, inclusive="both")
        & enriched["arr_usd"].notna()
        & (enriched["arr_usd"] > 50_000_000)
    ].copy()

    ai = ai[[
        "article_id", "title", "company_name", "published_date_normalized",
        "category_standardized", "arr_usd", "summary", "url", "industry",
        "founded_year", "headquarters", "employee_count", "is_public",
        "stock_ticker", "company_age", "company_size_category", "embedding",
        "top_similar_articles",
    ]]
    ai.columns = [
        "article_id", "title", "company_name", "published_date", "category",
        "arr_usd", "summary", "url", "industry", "founded_year", "headquarters",
        "employee_count", "is_public", "stock_ticker", "company_age",
        "company_size_category", "embedding", "top_similar_articles",
    ]
    ai.to_csv(out / "ai_articles_enriched.csv", index=False)

    unmatched = enriched[~enriched["company_match_flag"]][[
        "article_id", "source_row_number", "company_name", "company_resolution_status"
    ]]
    unmatched.to_csv(out / "unmatched_companies.csv", index=False)

    invalid_dates = enriched[enriched["date_parse_status"] != "parsed"][[
        "article_id", "source_row_number", "published_date", "date_parse_status"
    ]]
    invalid_dates.to_csv(out / "invalid_dates.csv", index=False)

    # Optional local DuckDB.
    write_duckdb({
        "dim_company": dim_company,
        "fct_article": fct_article,
        "fct_arr_observation": fct_arr,
    }, out / "yipitdata.duckdb")

    return {
        "dim_company": dim_company,
        "fct_article": fct_article,
        "fct_arr_observation": fct_arr,
        "ai_articles_enriched": ai,
        "unmatched_companies": unmatched,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--news", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--output-dir", default="outputs")
    args = p.parse_args()
    run_pipeline(args.news, args.metadata, args.output_dir)
