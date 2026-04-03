# Menu text classification dataset (NYCU AI HW1)

GitHub dataset documentation for a text classification task:
restaurant menu line items -> **13 standardized categories + `Other`**.

## What is included

Repository folders:

- `data/`
  - `menu_items_train.csv` (flattened menu items from annotated JSON)
  - `menu_items_train_cleaned.csv` (training-ready subset + cleaned text fields)
- `scripts/`
  - `build_menu_dataset.py` (optional) JSON -> flattened CSV
  - `category_standardization.py` category mapping: raw label -> standardized bucket
  - `train_menu_classifier.py` train/evaluate models with TF-IDF char n-grams
  - `menu_scraping.py` (optional) SerpAPI -> download menu photos for Document AI annotation
- `results/`
  - evaluation CSV outputs and (optionally) `confusion_matrix_run<N>.png`
- `requirements.txt`

Raw menu images downloaded by scraping are **not** committed to the repo.

## Dataset schema (`data/menu_items_train_cleaned.csv`)

Each row is one menu line item with these columns:

- `source_file` : source JSON filename
- `category_name_original` : raw category label from annotation export
- `category_name_english` : English category label (if present)
- `item_name_original` : item name (original language)
- `item_name_english` : item name (English, if present)
- `item_description_original` : item description (optional)
- `item_price` : price string (optional)
- `item_additional_notes` : extra notes (optional)

## Labels: 13 categories + `Other`

The model target `category_standardized` is created by `scripts/category_standardization.py`.

Implementation detail:

- `EXACT_MAP` applies exact string overrides.
- `RULES` is an ordered keyword matcher (first match wins).

The standardized labels are the 12 food/drink buckets plus `Other`, totaling **13 labels**.

## Reproducibility / how to run experiments

### Setup

```bash
pip install -r requirements.txt
```

### Train + evaluate (Run N)

```bash
python scripts/train_menu_classifier.py --run 1
python scripts/train_menu_classifier.py --run 2
```

By default, training uses:
`data/menu_items_train_cleaned.csv`.

Outputs (by default, saved under `results/`):

- `evaluation_summary_<run>.csv`
- `evaluation_per_class_<run>.csv`
- `evaluation_class_distribution_<run>.csv`
- (optional) `confusion_matrix_run<run>.png` (Logistic Regression only, unless `--no-confusion-heatmap`)

### Feature extraction used

`train_menu_classifier.py` builds `text_features` from:
`item_name_original + item_name_english`,
then uses `TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))`.

### Evaluation

Stratified 5-fold cross-validation (`StratifiedKFold`) with:
accuracy, macro F1, weighted F1,
plus per-class precision/recall/F1 and a confusion matrix.

## Optional: rebuild CSV from JSON

If you have the raw annotated JSON locally, you can regenerate the flattened CSV:

```bash
python scripts/build_menu_dataset.py cemta_menu_gcs/train -o data/menu_items_train.csv
```

Note: the exact input folder name depends on where the JSON is stored locally.

## Optional: menu photo scraping (for collecting images)

`scripts/menu_scraping.py` is a DATA COLLECTION helper:

- downloads menu-related photos from Google Maps via SerpAPI
- stores them in a local folder (default `Menu_Images/`)
- you upload those images to Document AI to produce labeled JSON
- the JSON is then flattened via `build_menu_dataset.py`

Security note:
- set `SERPAPI_API_KEY` as an environment variable
- do not commit API keys

