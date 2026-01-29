# California Coffee Market Expansion (PDR Implementation)

This project implements the **Location Identification Engine** and **Latte Price Predictive Model** described in `PDR.pdf`.

## What it does

- **Location identification / ranking**
  - Scores candidate California neighborhoods using:
    - Competition density (independent vs chain within ~2 miles, modeled here as counts)
    - Foot traffic proxies (transit / university / coworking proximity scores)
    - Growth velocity (YoY rent growth proxy)
    - Cost proxy (rent level)
- **Latte price prediction**
  - Trains a regression model to predict a **12oz latte price** from:
    - Median household income
    - Education level (% Bachelor's+)
    - Age demographic (% age 20–40)
    - Local rent index

Because API keys and scraping targets are not included in the PDR, the project supports:
- A **demo mode** that generates a realistic synthetic dataset
- **CSV input** for your own collected data
- Stubs where you can later plug in Yelp/Google Places, Census/ACS, Zillow/LoopNet, etc.

## Quick start (Windows PowerShell)

```powershell
cd "c:\Users\Admin\Downloads\vap sem 4\project in lecture"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.main --demo
```

This will:
- generate `data/demo_neighborhoods.csv`
- train the latte price model
- cluster neighborhoods into hotspots
- output a ranked top-5 list (and save artifacts under `artifacts/`)

## Run on your own CSV

Put a CSV like `data/demo_neighborhoods.csv` and run:

```powershell
python -m src.main --input "data\your_data.csv"
```

## Optional: Streamlit dashboard

```powershell
streamlit run src/app_streamlit.py
```

## Project structure

- `src/`
  - `main.py`: CLI entrypoint
  - `pipeline.py`: end-to-end workflow
  - `data_demo.py`: demo dataset generator
  - `models.py`: clustering + regression
  - `scoring.py`: weighted scoring for ranking
- `data/`: input datasets
- `artifacts/`: trained models, plots, outputs

## Data schema (CSV)

Required columns:
- `name` (string)
- `city` (string)
- `median_income` (number)
- `pct_bachelors_plus` (0-100)
- `pct_age_20_40` (0-100)
- `rent_index` (number)
- `rent_yoy_growth` (number, e.g. 0.08 for 8%)
- `independent_coffee_count_2mi` (int)
- `chain_coffee_count_2mi` (int)
- `transit_score` (0-100)
- `university_score` (0-100)
- `coworking_score` (0-100)

## Notes

- The PDR success metric (“within $0.25 of market leaders”) requires real observed pricing labels for training/validation.
  In demo mode, labels are generated from a controlled formula with noise.

# Location-Predictor-vap-01
# VAP_Sem_04_Project_01
# VAP_Sem_04_Project_01
# Caffeina-Analytics
