
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pipeline import (
    clean_and_enrich_articles,
    parse_revenue,
    normalize_date,
    resolve_company,
    standardize_category,
)

META = {"Amazon Web Services": {"founded_year": 2006}}

def test_usd_billion():
    assert parse_revenue("$5.2B")[0] == 5_200_000_000

def test_full_number():
    assert parse_revenue("$5,200,000,000")[0] == 5_200_000_000

def test_words():
    assert parse_revenue("5.2 billion")[0] == 5_200_000_000

def test_eur_conversion():
    assert parse_revenue("€2.1B")[0] == 2_310_000_000

def test_gbp_conversion():
    assert parse_revenue("£100M")[0] == 127_000_000

def test_jpy_conversion():
    assert parse_revenue("¥150,000,000,000")[0] == 1_000_000_000

def test_range_midpoint():
    assert parse_revenue("$10M - $20M")[0] == 15_000_000

def test_missing_revenue_is_not_arr():
    assert parse_revenue("Not disclosed")[0] is None
    assert parse_revenue("N/A")[0] is None

def test_us_date():
    assert str(normalize_date("02/23/2023")[0].date()) == "2023-02-23"

def test_eu_date():
    assert str(normalize_date("23-08-2023")[0].date()) == "2023-08-23"

def test_iso_date():
    assert str(normalize_date("2023-12-13")[0].date()) == "2023-12-13"

def test_category():
    assert standardize_category("Artificial Intelligence")[0] == "AI_ML"

def test_alias():
    assert resolve_company("AWS", META) == ("Amazon Web Services", "alias")


def test_unmatched_company_is_handled_without_key_error(tmp_path):
    news_path = tmp_path / "news.csv"
    metadata_path = tmp_path / "company_metadata.json"

    news_path.write_text(
        "article_id,company_name,published_date,revenue,category,title,summary,url,author,word_count\n"
        "ART999,UnknownCo,2024-02-01,$10M,AI/ML,Test Title,Test Summary,http://example.com,Author,120\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps({
            "Microsoft": {
                "industry": "AI/ML",
                "founded_year": 1975,
                "headquarters": "Redmond, WA",
                "employee_count": 100000,
                "is_public": True,
                "stock_ticker": "MSFT",
            }
        }),
        encoding="utf-8",
    )

    enriched, _ = clean_and_enrich_articles(news_path, metadata_path)

    assert not bool(enriched.loc[0, "company_match_flag"])
    assert pd.isna(enriched.loc[0, "company_id"])
    assert enriched.loc[0, "company_size_category"] == "UNKNOWN"
