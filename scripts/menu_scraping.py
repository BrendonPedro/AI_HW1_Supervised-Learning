"""
Menu image scraping with SerpAPI (Google Maps).

This script is for the DATA COLLECTION step only.
It downloads menu-related photos into a local folder (e.g. `Menu_Images/`),
which you then upload to Document AI for manual annotation.

Requirements:
  - `pip install requests google-search-results serpapi` (package `google-search-results` provides `serpapi`)
  - Set env var `SERPAPI_API_KEY`

IMPORTANT:
  - Do NOT commit API keys. This script reads `SERPAPI_API_KEY` from your environment.
  - Raw images can be large; the repo typically submits the derived CSV dataset, not the image folder.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import requests
from serpapi import GoogleSearch


def sanitize_filename(s: str) -> str:
    s = s.strip()
    # Replace path separators and other unsafe characters.
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", s)
    return s[:120] if len(s) > 120 else s


def scrape_menu_photos(
    *,
    api_key: str,
    query: str,
    location: str,
    category_id: str,
    offsets: list[int],
    max_menus_per_restaurant: int,
    restaurant_photo_query: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for o in offsets:
        print(f"Starting from offset={o}")
        rest_params = {
            "engine": "google_maps",
            "q": query,
            "ll": location,
            "type": "search",
            "google_domain": "google.com",
            "hl": "en",
            "gl": "us",
            "api_key": api_key,
            "start": o,
        }

        rest_search = GoogleSearch(rest_params)
        rest_results = rest_search.get_dict()
        local_results = rest_results.get("local_results", []) or []

        rest_data = [(r.get("title", ""), r.get("data_id")) for r in local_results]
        rest_data = [(t, d) for (t, d) in rest_data if t and d]

        for restaurant_title, data_id in rest_data:
            print(f"Scraping menus from: {restaurant_title}")
            menu_params = {
                "engine": "google_maps_photos",
                "data_id": data_id,
                "q": restaurant_photo_query,
                "hl": "en",
                "category_id": category_id,
                "api_key": api_key,
            }
            menu_search = GoogleSearch(menu_params)
            menu_results = menu_search.get_dict()

            photos = menu_results.get("photos", []) or []
            i = 0
            for p in photos:
                if i >= max_menus_per_restaurant:
                    break
                img_url = p.get("image")
                if not img_url:
                    continue
                try:
                    img_data = requests.get(img_url, timeout=20).content
                except Exception as exc:
                    print(f"  - Failed downloading photo {i} ({restaurant_title}): {exc}")
                    continue

                safe_title = sanitize_filename(restaurant_title)
                out_path = output_dir / f"{safe_title}_{i}.jpg"
                try:
                    out_path.write_bytes(img_data)
                except Exception as exc:
                    print(f"  - Failed writing photo {i} ({out_path}): {exc}")
                    continue

                i += 1

            print(f"Finished scraping menus from: {restaurant_title} (saved {i} photos)")


def main() -> None:
    p = argparse.ArgumentParser(description="Scrape menu photos with SerpAPI (Google Maps).")
    p.add_argument("--query", default="restaurants in Toufen, Toufen City, Miaoli County")
    p.add_argument("--location", default="@24.6907758,120.8972858,17z")
    p.add_argument("--category-id", default="CgIYIQ", help="Google Maps photos category_id (SerpAPI).")
    p.add_argument(
        "--offsets",
        nargs="*",
        type=int,
        default=[0, 20, 40, 60, 80, 100],
        help="Offsets for pagination (SerpAPI `start`).",
    )
    p.add_argument("--max-menus", type=int, default=5, help="Max photos per restaurant.")
    p.add_argument("--restaurant-photo-query", default="Coffee", help="Search query used for photo search.")
    p.add_argument(
        "--output-dir",
        type=str,
        default="Menu_Images",
        help="Local folder to save downloaded photos.",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default="",
        help="(Optional) SerpAPI API key. Prefer env var SERPAPI_API_KEY and leave this empty.",
    )
    args = p.parse_args()

    api_key = args.api_key or os.environ.get("SERPAPI_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "Missing SerpAPI key. Set env var SERPAPI_API_KEY or pass --api-key."
        )

    scrape_menu_photos(
        api_key=api_key,
        query=args.query,
        location=args.location,
        category_id=args.category_id,
        offsets=args.offsets,
        max_menus_per_restaurant=args.max_menus,
        restaurant_photo_query=args.restaurant_photo_query,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

