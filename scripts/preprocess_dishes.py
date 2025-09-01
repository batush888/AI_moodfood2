#!/usr/bin/env python3
"""
Preprocess Shanghai Gold dishes CSV into JSONL and Parquet datasets.

Input CSV (data/shanghai_gold_200_dishes.csv) headers:
 - dish_name
 - calories_kcal
 - nutrients
 - ingredients
 - user_inputs

Outputs:
 - data/processed/dish_dataset.jsonl
 - data/processed/dish_dataset.parquet

The script is robust to minor formatting differences and missing values.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Taxonomy mapping for contexts
# -----------------------------
TAXONOMY_TAG_MAP = {
    # Mood / comfort
    "comfort": "COMFORT_EMOTIONAL",
    "cozy": "COMFORT_EMOTIONAL",
    "warm": "COMFORT_EMOTIONAL",
    "nostalgic": "COMFORT_EMOTIONAL",
    "homestyle": "COMFORT_EMOTIONAL",
    "hearty": "COMFORT_EMOTIONAL",
    "soothing": "COMFORT_EMOTIONAL",
    "relax": "COMFORT_EMOTIONAL",
    "relaxing": "COMFORT_EMOTIONAL",
    # Energy / vitality
    "energized": "ENERGY_VITALITY",
    "energy": "ENERGY_VITALITY",
    "recovery": "ENERGY_VITALITY",
    "post-workout": "ENERGY_VITALITY",
    "preworkout": "ENERGY_VITALITY",
    "gym": "ENERGY_VITALITY",
    "stamina": "ENERGY_VITALITY",
    "boost": "ENERGY_VITALITY",
    # Adventure / novelty
    "adventurous": "EXPERIMENTAL_FLAVOR",
    "novel": "EXPERIMENTAL_FLAVOR",
    "spicy": "EXPERIMENTAL_FLAVOR",
    "bold": "EXPERIMENTAL_FLAVOR",
    "chili": "EXPERIMENTAL_FLAVOR",
    "peppery": "EXPERIMENTAL_FLAVOR",
    "exotic": "EXPERIMENTAL_FLAVOR",
    "unique": "EXPERIMENTAL_FLAVOR",
    # Health / light
    "healthy": "HEALTH_LIGHT",
    "light": "HEALTH_LIGHT",
    "low-calorie": "HEALTH_LIGHT",
    "low calorie": "HEALTH_LIGHT",
    "refreshing": "HEALTH_LIGHT",
    "clean": "HEALTH_LIGHT",
    "low-fat": "HEALTH_LIGHT",
    "low sugar": "HEALTH_LIGHT",
    "detox": "HEALTH_LIGHT",
    # Weather / season
    "hot": "WEATHER_HOT",
    "cold": "WEATHER_COLD",
    "sunny": "WEATHER_HOT",
    "summer": "WEATHER_HOT",
    "humid": "WEATHER_HOT",
    "heat": "WEATHER_HOT",
    "chilly": "WEATHER_COLD",
    "winter": "WEATHER_COLD",
    "cool": "WEATHER_COLD",
    "rain": "WEATHER_RAIN",
    "rainy": "WEATHER_RAINY",
    "drizzle": "WEATHER_RAIN",
    "stormy": "WEATHER_RAIN",
    # Social / occasion
    "date": "SOCIAL_ROMANTIC",
    "friends": "SOCIAL_GROUP",
    "family": "SOCIAL_FAMILY",
    "romantic": "SOCIAL_ROMANTIC",
    "date-night": "SOCIAL_ROMANTIC",
    "partner": "SOCIAL_ROMANTIC",
    "lover": "SOCIAL_ROMANTIC",
    "hangout": "SOCIAL_GROUP",
    "group": "SOCIAL_GROUP",
    "party": "SOCIAL_GROUP",
    "meetup": "SOCIAL_GROUP",
    "kids": "SOCIAL_FAMILY",
    "kid-friendly": "SOCIAL_FAMILY",
}


NUTRIENT_KEYS = ("protein", "fat", "carbs", "fiber")


def _ensure_processed_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def parse_nutrients(value: str | float | int | None) -> Dict[str, float]:
    """Parse free-form nutrient string to a normalized dict.

    Accepts formats like:
      - "protein:21g, fat:10g, carbs:38g, fiber:3g"
      - "P21 F10 C38 Fi3"
      - "protein=21;fat=10;carbs=38;fiber=3"
    Returns zeros for missing values.
    """
    result = {k: 0.0 for k in NUTRIENT_KEYS}
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return result
    if not isinstance(value, str):
        try:
            # If it's already a dict-like JSON string from CSV
            return json.loads(str(value))
        except Exception:
            return result

    text = value.strip()
    if not text:
        return result

    # Common separators
    parts = re.split(r"[;,|]\s*", text)
    # Regex to capture key[:= ]? value with optional units
    pattern = re.compile(r"(protein|prot|p|fat|f|carbs|carb|c|fiber|fi|fib)\s*[:=]?\s*([\d.]+)")
    found = False
    for part in parts:
        for m in pattern.finditer(part.lower()):
            key, val = m.groups()
            key_norm = (
                "protein" if key in {"protein", "prot", "p"}
                else "fat" if key in {"fat", "f"}
                else "carbs" if key in {"carbs", "carb", "c"}
                else "fiber"
            )
            try:
                result[key_norm] = float(val)
                found = True
            except ValueError:
                pass

    # Try JSON dict if pattern fails
    if not found:
        try:
            as_dict = json.loads(text)
            for k in NUTRIENT_KEYS:
                if k in as_dict:
                    result[k] = float(as_dict[k])
        except Exception:
            pass

    return result


def parse_ingredients(value: str | None) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    # Split on commas/semicolons or pipes
    items = re.split(r"[,;|]", text)
    return [i.strip().lower() for i in items if i.strip()]


def parse_context_tags(value: str | None) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    raw_tags = re.split(r"[,;|]", text)
    tags: List[str] = []
    for t in raw_tags:
        tag = t.strip().lower()
        if not tag:
            continue
        mapped = TAXONOMY_TAG_MAP.get(tag)
        if mapped:
            tags.append(mapped)
    # Deduplicate, preserve order
    seen = set()
    unique_tags = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique_tags.append(t)
    return unique_tags


def normalize_row(row: pd.Series) -> Dict[str, object]:
    dish_name = str(row.get("dish_name", "")).strip()
    # calories may be numeric or string; coerce to int
    calories_raw = row.get("calories_kcal", None)
    calories: Optional[int] = None
    try:
        if calories_raw is not None and not (isinstance(calories_raw, float) and pd.isna(calories_raw)):
            calories = int(float(calories_raw))
    except Exception:
        calories = None

    nutrients = parse_nutrients(row.get("nutrients"))
    ingredients = parse_ingredients(row.get("ingredients"))
    contexts = parse_context_tags(row.get("user_inputs"))

    record = {
        "dish_name": dish_name,
        "calories": calories if calories is not None else 0,
        "nutrients": nutrients,
        "ingredients": ingredients,
        "contexts": contexts,
    }
    return record


def save_jsonl(records: List[Dict[str, object]], path: str) -> None:
    _ensure_processed_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_parquet(records: List[Dict[str, object]], path: str) -> None:
    _ensure_processed_dir(path)
    df = pd.DataFrame(records)
    # Ensure complex/object columns are stringified for Parquet compatibility when needed
    for col in ("nutrients", "ingredients", "contexts"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
    try:
        df.to_parquet(path, engine="pyarrow", index=False)
    except Exception:
        # Fallback to fastparquet if pyarrow missing
        df.to_parquet(path, engine="fastparquet", index=False)


def run(input_csv: str, jsonl_out: str, parquet_out: str) -> Tuple[int, str, str]:
    df = pd.read_csv(input_csv)
    required_cols = ["dish_name", "calories_kcal", "nutrients", "ingredients", "user_inputs"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    records = [normalize_row(row) for _, row in df.iterrows()]
    save_jsonl(records, jsonl_out)
    save_parquet(records, parquet_out)
    return len(records), jsonl_out, parquet_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess dishes CSV into JSONL and Parquet.")
    parser.add_argument("--input", default="data/shanghai_gold_200_dishes.csv", help="Path to input CSV")
    parser.add_argument("--jsonl", default="data/processed/dish_dataset.jsonl", help="Output JSONL path")
    parser.add_argument("--parquet", default="data/processed/dish_dataset.parquet", help="Output Parquet path")
    args = parser.parse_args()

    count, jsonl_path, parquet_path = run(args.input, args.jsonl, args.parquet)
    print(json.dumps({
        "records": count,
        "jsonl": os.path.abspath(jsonl_path),
        "parquet": os.path.abspath(parquet_path),
    }, indent=2))


if __name__ == "__main__":
    main()


