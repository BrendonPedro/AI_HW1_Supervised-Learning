from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_first_property_text(entity: Dict[str, Any], prop_type: str) -> Optional[str]:
    """Return the first property mentionText matching prop_type."""
    for prop in entity.get("properties", []):
        if prop.get("type") == prop_type:
            text = prop.get("mentionText")
            if text:
                return text.strip()
    return None


def clean_text(value: Optional[str]) -> str:
    """Normalize whitespace and handle None safely."""
    if not value:
        return ""
    return " ".join(value.replace("\n", " ").replace("|", " ").split()).strip()


def extract_rows_from_json(data: Dict[str, Any], source_file: str) -> List[Dict[str, str]]:
    """
    Extract flat menu-item rows from a Document AI JSON.

    Expected labels include:
    - category
      - category_name_original
      - category_name_english
    - items
      - item_name_original
      - item_name_english
      - item_description_original
      - item_price
      - item_additional_notes
    """
    rows: List[Dict[str, str]] = []

    # Document AI exports categories and items in a single sequence.
    # We keep the latest seen category so subsequent item rows inherit it.
    current_category_original = ""
    current_category_english = ""

    entities = data.get("entities", [])

    for entity in entities:
        entity_type = entity.get("type", "")

        if entity_type == "category":
            current_category_original = clean_text(
                get_first_property_text(entity, "category_name_original")
                or entity.get("mentionText")
            )
            current_category_english = clean_text(
                get_first_property_text(entity, "category_name_english")
            )

        elif entity_type == "items":
            item_name_original = clean_text(
                get_first_property_text(entity, "item_name_original")
            )
            item_name_english = clean_text(
                get_first_property_text(entity, "item_name_english")
            )
            item_description_original = clean_text(
                get_first_property_text(entity, "item_description_original")
            )
            item_price = clean_text(
                get_first_property_text(entity, "item_price")
            )
            item_additional_notes = clean_text(
                get_first_property_text(entity, "item_additional_notes")
            )

            # Skip obviously broken rows with no usable name
            if not item_name_original and not item_name_english:
                continue

            rows.append(
                {
                    "source_file": source_file,
                    "category_name_original": current_category_original,
                    "category_name_english": current_category_english,
                    "item_name_original": item_name_original,
                    "item_name_english": item_name_english,
                    "item_description_original": item_description_original,
                    "item_price": item_price,
                    "item_additional_notes": item_additional_notes,
                }
            )

    return rows


def build_dataset(input_dir: str, output_csv: str) -> None:
    """Walk through JSON files and build a flat CSV dataset."""
    input_path = Path(input_dir)
    all_rows: List[Dict[str, str]] = []

    # Deterministic ordering keeps row generation stable across runs.
    for json_file in sorted(input_path.rglob("*.json")):
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            rows = extract_rows_from_json(data, source_file=str(json_file.name))
            all_rows.extend(rows)
        except Exception as exc:
            print(f"Failed to process {json_file}: {exc}")

    fieldnames = [
        "source_file",
        "category_name_original",
        "category_name_english",
        "item_name_original",
        "item_name_english",
        "item_description_original",
        "item_price",
        "item_additional_notes",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved {len(all_rows)} rows to {output_csv}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Flatten Document AI menu JSON into CSV.")
    p.add_argument(
        "input_dir",
        nargs="?",
        default="train",
        help="Folder containing JSON (recursive). Default: train",
    )
    p.add_argument(
        "-o",
        "--output",
        default="data/menu_items_dataset.csv",
        help="Output CSV path. Default: data/menu_items_dataset.csv",
    )
    args = p.parse_args()
    build_dataset(input_dir=args.input_dir, output_csv=args.output)