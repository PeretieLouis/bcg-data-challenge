"""
Bronze to Silver Data Processing Pipeline
==========================================
Transforms raw data (bronze) into cleaned, validated data (silver).

Input: data/bronze/
Output: data/silver/
    - barley_yield.parquet (cleaned yield data, 89 departments)
    - climate_historical.parquet (historical climate, pivoted metrics)
    - climate_ssp1_2_6.parquet (scenario 1 climate data)
    - climate_ssp2_4_5.parquet (scenario 2 climate data) [NOTE: 8 depts missing 2 metrics]
    - climate_ssp5_8_5.parquet (scenario 3 climate data)
"""

from pathlib import Path

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
BRONZE_PATH = PROJECT_ROOT / "data" / "bronze"
SILVER_PATH = PROJECT_ROOT / "data" / "silver"

# Departments to drop (not in climate data or have no yield data)
DEPARTMENTS_TO_DROP = [
    "Corse_du_Sud",
    "Haute_Corse",
    "Hauts_de_Seine",
    "Paris",
    "Seine_Saint_Denis",
    "Seine_SeineOise",
    "Val_d_Oise",
    "Val_de_Marne",
]

# SSP2_4_5 departments with missing metrics (flagged for later decision)
SSP2_DEPARTMENTS_WITH_MISSING_METRICS = [
    "Calvados",
    "Deux_Sevres",
    "Essonne",
    "Eure",
    "Rhone",
    "Tarn_et_Garonne",
    "Territoire_de_Belfort",
    "Vaucluse",
]


def load_bronze_data() -> dict[str, pd.DataFrame]:
    """Load all bronze (raw) data files."""
    data = {}

    # Load barley yield data (separator is ;)
    barley_path = BRONZE_PATH / "barley_yield_from_1982.csv"
    if barley_path.exists():
        data["barley_yield"] = pd.read_csv(barley_path, sep=";")
        print(f"  ✓ Loaded barley_yield: {data['barley_yield'].shape}")

    # Load climate data
    climate_path = BRONZE_PATH / "climate_data_from_1982.parquet"
    if climate_path.exists():
        data["climate"] = pd.read_parquet(climate_path)
        print(f"  ✓ Loaded climate: {data['climate'].shape}")

    return data


def clean_barley_yield(df: pd.DataFrame, valid_departments: set) -> pd.DataFrame:
    """
    Clean barley yield dataset.

    Steps:
    1. Drop the first column (Unnamed: 0 - just row numbers)
    2. Drop departments not in climate data
    3. Calculate missing yields from production/area
    4. Rename 'department' to 'nom_dep' for consistency
    """
    df_clean = df.copy()

    # Step 1: Drop the index column
    if "Unnamed: 0" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Unnamed: 0"])
        print("    → Dropped 'Unnamed: 0' column")

    # Step 2: Filter to only departments present in climate data
    initial_depts = df_clean["department"].nunique()
    df_clean = df_clean[df_clean["department"].isin(valid_departments)]
    final_depts = df_clean["department"].nunique()
    print(f"    → Filtered departments: {initial_depts} → {final_depts}")

    # Step 3: Calculate missing yields from production / area
    missing_yields_before = df_clean["yield"].isnull().sum()
    mask = df_clean["yield"].isnull() & df_clean["production"].notna() & df_clean["area"].notna()
    df_clean.loc[mask, "yield"] = df_clean.loc[mask, "production"] / df_clean.loc[mask, "area"]
    missing_yields_after = df_clean["yield"].isnull().sum()
    print(f"    → Filled missing yields: {missing_yields_before} → {missing_yields_after}")

    # Step 4: Drop rows with remaining missing yields (cannot be recovered)
    rows_before = len(df_clean)
    df_clean = df_clean.dropna(subset=["yield"])
    rows_dropped = rows_before - len(df_clean)
    if rows_dropped > 0:
        print(f"    → Dropped {rows_dropped} rows with unrecoverable missing yields")

    # Step 5: Rename department column for consistency with climate data
    df_clean = df_clean.rename(columns={"department": "nom_dep"})
    print("    → Renamed 'department' to 'nom_dep'")

    # Reset index
    df_clean = df_clean.reset_index(drop=True)

    return df_clean


def pivot_climate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot climate data so each metric becomes a column.

    Input columns: scenario, nom_dep, code_dep, time, year, metric, value
    Output columns: scenario, nom_dep, code_dep, time, year,
                    near_surface_air_temperature,
                    daily_maximum_near_surface_air_temperature,
                    precipitation
    """
    # Pivot the metric column to separate columns
    df_pivoted = df.pivot_table(
        index=["scenario", "nom_dep", "code_dep", "time", "year"],
        columns="metric",
        values="value",
        aggfunc="first",  # Should be unique anyway
    ).reset_index()

    # Flatten column names
    df_pivoted.columns.name = None

    return df_pivoted


def clean_climate_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Clean and split climate dataset.

    Steps:
    1. Pivot metrics to columns
    2. Split by scenario into 4 datasets:
       - historical (1982-2014) - for training/testing
       - ssp1_2_6 (2015-2050) - scenario 1
       - ssp2_4_5 (2015-2050) - scenario 2 [flagged: 8 depts missing 2 metrics]
       - ssp5_8_5 (2015-2050) - scenario 3

    Returns dict of DataFrames, one per scenario.
    """
    climate_datasets = {}

    scenarios = df["scenario"].unique()
    print(f"    → Found {len(scenarios)} scenarios: {list(scenarios)}")

    for scenario in scenarios:
        df_scenario = df[df["scenario"] == scenario].copy()

        # Pivot metrics to columns
        df_pivoted = pivot_climate_metrics(df_scenario)

        # Drop the scenario column (redundant now that we're splitting)
        df_pivoted = df_pivoted.drop(columns=["scenario"])

        # Sort by department and time
        df_pivoted = df_pivoted.sort_values(["nom_dep", "time"]).reset_index(drop=True)

        # Add flag for ssp2_4_5 missing data
        if scenario == "ssp2_4_5":
            missing_count = (
                df_pivoted[df_pivoted["nom_dep"].isin(SSP2_DEPARTMENTS_WITH_MISSING_METRICS)][
                    "daily_maximum_near_surface_air_temperature"
                ]
                .isnull()
                .sum()
            )
            print(f"    ⚠ {scenario}: {missing_count} rows with missing metrics (flagged)")

        year_range = f"{df_pivoted['year'].min()}-{df_pivoted['year'].max()}"
        print(f"    → {scenario}: {df_pivoted.shape} | years: {year_range}")

        climate_datasets[f"climate_{scenario}"] = df_pivoted

    return climate_datasets


def validate_silver_data(data: dict[str, pd.DataFrame]) -> bool:
    """Validate silver data quality."""
    print("  Validating datasets...")
    all_valid = True

    for name, df in data.items():
        issues = []

        # Check for empty dataframe
        if len(df) == 0:
            issues.append("DataFrame is empty")

        # Check for nom_dep column
        if "nom_dep" not in df.columns:
            issues.append("Missing 'nom_dep' column")

        # Check for year column
        if "year" not in df.columns:
            issues.append("Missing 'year' column")

        if issues:
            print(f"  ✗ {name}: {', '.join(issues)}")
            all_valid = False
        else:
            print(f"  ✓ {name}: OK ({len(df)} rows, {len(df.columns)} cols)")

    return all_valid


def save_silver_data(data: dict[str, pd.DataFrame]) -> None:
    """Save cleaned data to silver layer."""
    SILVER_PATH.mkdir(parents=True, exist_ok=True)

    for name, df in data.items():
        output_path = SILVER_PATH / f"{name}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"  ✓ Saved {name}.parquet ({len(df):,} rows)")


def run_bronze_to_silver() -> dict[str, pd.DataFrame]:
    """Main pipeline: Bronze → Silver."""
    print("=" * 60)
    print("BRONZE → SILVER PIPELINE")
    print("=" * 60)

    # Load raw data
    print("\n[1/5] Loading bronze data...")
    bronze_data = load_bronze_data()

    # Get valid departments from climate data
    print("\n[2/5] Identifying valid departments...")
    climate_departments = set(bronze_data["climate"]["nom_dep"].unique())
    print(f"  → {len(climate_departments)} departments in climate data")
    print(f"  → Dropping {len(DEPARTMENTS_TO_DROP)} departments from yield data")

    # Clean barley yield
    print("\n[3/5] Cleaning barley yield data...")
    silver_data = {}
    silver_data["barley_yield"] = clean_barley_yield(
        bronze_data["barley_yield"], climate_departments
    )

    # Clean and split climate data
    print("\n[4/5] Cleaning and splitting climate data...")
    climate_datasets = clean_climate_data(bronze_data["climate"])
    silver_data.update(climate_datasets)

    # Validate
    print("\n[5/5] Validating silver data...")
    if not validate_silver_data(silver_data):
        raise ValueError("Silver data validation failed!")

    # Save
    print("\n[6/6] Saving silver data...")
    save_silver_data(silver_data)

    print("\n" + "=" * 60)
    print("BRONZE → SILVER COMPLETE")
    print("=" * 60)
    print(f"\nOutput files in: {SILVER_PATH}")
    print("  - barley_yield.parquet")
    print("  - climate_historical.parquet (training/test data)")
    print("  - climate_ssp1_2_6.parquet (scenario 1 predictions)")
    print("  - climate_ssp2_4_5.parquet (scenario 2 predictions) ⚠ flagged")
    print("  - climate_ssp5_8_5.parquet (scenario 3 predictions)")

    return silver_data


if __name__ == "__main__":
    run_bronze_to_silver()
