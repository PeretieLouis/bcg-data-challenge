"""
Silver to Gold Data Processing Pipeline
========================================
Transforms cleaned data (silver) into feature-engineered data (gold).

Input: data/silver/
Output: data/gold/
    - training_data.parquet (historical climate + yield, 1982-2014, yearly features)
    - climate_ssp1_2_6_features.parquet (scenario 1, yearly features for prediction)
    - climate_ssp2_4_5_features.parquet (scenario 2, yearly features for prediction)
    - climate_ssp5_8_5_features.parquet (scenario 3, yearly features for prediction)
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SILVER_PATH = PROJECT_ROOT / "data" / "silver"
GOLD_PATH = PROJECT_ROOT / "data" / "gold"

# Training data year range (historical climate data availability)
TRAIN_YEAR_START = 1982
TRAIN_YEAR_END = 2014

# Season definitions (month ranges)
SEASONS = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
}

# Base temperature for Growing Degree Days (Celsius converted from Kelvin)
GDD_BASE_TEMP_KELVIN = 273.15  # 0C in Kelvin

# French department geographic data (approximate centroids and mean altitudes)
# Sources: INSEE, IGN
DEPARTMENT_GEO = {
    "Ain": {"lat": 46.07, "lon": 5.35, "altitude": 450},
    "Aisne": {"lat": 49.45, "lon": 3.62, "altitude": 120},
    "Allier": {"lat": 46.37, "lon": 3.17, "altitude": 350},
    "Alpes_de_Haute_Provence": {"lat": 44.10, "lon": 6.24, "altitude": 1200},
    "Alpes_Maritimes": {"lat": 43.93, "lon": 7.12, "altitude": 800},
    "Ardeche": {"lat": 44.75, "lon": 4.42, "altitude": 700},
    "Ardennes": {"lat": 49.62, "lon": 4.63, "altitude": 250},
    "Ariege": {"lat": 42.93, "lon": 1.50, "altitude": 900},
    "Aube": {"lat": 48.30, "lon": 4.08, "altitude": 150},
    "Aude": {"lat": 43.10, "lon": 2.40, "altitude": 400},
    "Aveyron": {"lat": 44.28, "lon": 2.67, "altitude": 700},
    "Bas_Rhin": {"lat": 48.67, "lon": 7.55, "altitude": 250},
    "Bouches_du_Rhone": {"lat": 43.55, "lon": 5.08, "altitude": 200},
    "Calvados": {"lat": 49.10, "lon": -0.37, "altitude": 100},
    "Cantal": {"lat": 45.05, "lon": 2.67, "altitude": 1000},
    "Charente": {"lat": 45.75, "lon": 0.17, "altitude": 150},
    "Charente_Maritime": {"lat": 45.75, "lon": -0.83, "altitude": 30},
    "Cher": {"lat": 47.07, "lon": 2.40, "altitude": 200},
    "Correze": {"lat": 45.37, "lon": 1.87, "altitude": 500},
    "Corse_du_Sud": {"lat": 41.87, "lon": 8.98, "altitude": 600},
    "Cote_d_Or": {"lat": 47.42, "lon": 4.77, "altitude": 400},
    "Cotes_d_Armor": {"lat": 48.45, "lon": -2.97, "altitude": 150},
    "Creuse": {"lat": 46.08, "lon": 2.03, "altitude": 500},
    "Deux_Sevres": {"lat": 46.55, "lon": -0.33, "altitude": 100},
    "Dordogne": {"lat": 45.13, "lon": 0.75, "altitude": 250},
    "Doubs": {"lat": 47.17, "lon": 6.35, "altitude": 600},
    "Drome": {"lat": 44.68, "lon": 5.17, "altitude": 500},
    "Essonne": {"lat": 48.52, "lon": 2.25, "altitude": 100},
    "Eure": {"lat": 49.10, "lon": 1.15, "altitude": 120},
    "Eure_et_Loir": {"lat": 48.30, "lon": 1.33, "altitude": 150},
    "Finistere": {"lat": 48.40, "lon": -4.05, "altitude": 100},
    "Gard": {"lat": 44.00, "lon": 4.08, "altitude": 300},
    "Gers": {"lat": 43.65, "lon": 0.58, "altitude": 200},
    "Gironde": {"lat": 44.83, "lon": -0.58, "altitude": 50},
    "Haute_Corse": {"lat": 42.42, "lon": 9.20, "altitude": 700},
    "Haute_Garonne": {"lat": 43.35, "lon": 1.17, "altitude": 300},
    "Haute_Loire": {"lat": 45.08, "lon": 3.88, "altitude": 900},
    "Haute_Marne": {"lat": 48.08, "lon": 5.25, "altitude": 350},
    "Haute_Saone": {"lat": 47.62, "lon": 6.08, "altitude": 350},
    "Haute_Savoie": {"lat": 46.00, "lon": 6.42, "altitude": 1200},
    "Haute_Vienne": {"lat": 45.88, "lon": 1.25, "altitude": 400},
    "Hautes_Alpes": {"lat": 44.67, "lon": 6.25, "altitude": 1500},
    "Hautes_Pyrenees": {"lat": 43.05, "lon": 0.17, "altitude": 900},
    # alternate spelling
    "Hautes_pyrenees": {"lat": 43.05, "lon": 0.17, "altitude": 900},
    "Haut_Rhin": {"lat": 47.87, "lon": 7.25, "altitude": 400},
    "Herault": {"lat": 43.58, "lon": 3.42, "altitude": 300},
    "Ille_et_Vilaine": {"lat": 48.12, "lon": -1.67, "altitude": 80},
    "Indre": {"lat": 46.78, "lon": 1.58, "altitude": 150},
    "Indre_et_Loire": {"lat": 47.25, "lon": 0.75, "altitude": 100},
    "Isere": {"lat": 45.25, "lon": 5.58, "altitude": 700},
    "Jura": {"lat": 46.75, "lon": 5.75, "altitude": 550},
    "Landes": {"lat": 43.92, "lon": -0.75, "altitude": 50},
    "Loir_et_Cher": {"lat": 47.58, "lon": 1.33, "altitude": 120},
    "Loire": {"lat": 45.75, "lon": 4.17, "altitude": 500},
    "Loire_Atlantique": {"lat": 47.33, "lon": -1.75, "altitude": 50},
    "Loiret": {"lat": 47.92, "lon": 2.17, "altitude": 130},
    "Lot": {"lat": 44.62, "lon": 1.67, "altitude": 350},
    "Lot_et_Garonne": {"lat": 44.35, "lon": 0.50, "altitude": 150},
    "Lozere": {"lat": 44.52, "lon": 3.50, "altitude": 1000},
    "Maine_et_Loire": {"lat": 47.42, "lon": -0.50, "altitude": 60},
    "Manche": {"lat": 49.08, "lon": -1.25, "altitude": 80},
    "Marne": {"lat": 48.95, "lon": 4.17, "altitude": 150},
    "Mayenne": {"lat": 48.15, "lon": -0.75, "altitude": 100},
    "Meurthe_et_Moselle": {"lat": 48.83, "lon": 6.17, "altitude": 300},
    "Meuse": {"lat": 49.00, "lon": 5.33, "altitude": 300},
    "Morbihan": {"lat": 47.75, "lon": -2.75, "altitude": 80},
    "Moselle": {"lat": 49.03, "lon": 6.67, "altitude": 280},
    "Nievre": {"lat": 47.12, "lon": 3.50, "altitude": 300},
    "Nord": {"lat": 50.43, "lon": 3.08, "altitude": 50},
    "Oise": {"lat": 49.42, "lon": 2.42, "altitude": 100},
    "Orne": {"lat": 48.62, "lon": 0.00, "altitude": 200},
    "Paris": {"lat": 48.87, "lon": 2.33, "altitude": 50},
    "Pas_de_Calais": {"lat": 50.50, "lon": 2.33, "altitude": 80},
    "Puy_de_Dome": {"lat": 45.72, "lon": 3.00, "altitude": 700},
    "Pyrenees_Atlantiques": {"lat": 43.25, "lon": -0.75, "altitude": 500},
    "Pyrenees_Orientales": {"lat": 42.60, "lon": 2.50, "altitude": 600},
    "Rhone": {"lat": 45.87, "lon": 4.63, "altitude": 350},
    "Saone_et_Loire": {"lat": 46.65, "lon": 4.50, "altitude": 350},
    "Sarthe": {"lat": 47.93, "lon": 0.17, "altitude": 80},
    "Savoie": {"lat": 45.50, "lon": 6.42, "altitude": 1400},
    "Seine_et_Marne": {"lat": 48.62, "lon": 2.92, "altitude": 100},
    "Seine_Maritime": {"lat": 49.67, "lon": 1.00, "altitude": 100},
    "Somme": {"lat": 49.92, "lon": 2.33, "altitude": 70},
    "Tarn": {"lat": 43.83, "lon": 2.17, "altitude": 400},
    "Tarn_et_Garonne": {"lat": 44.02, "lon": 1.28, "altitude": 200},
    "Territoire_de_Belfort": {"lat": 47.63, "lon": 6.92, "altitude": 450},
    "Val_d_Oise": {"lat": 49.08, "lon": 2.17, "altitude": 80},
    "Var": {"lat": 43.47, "lon": 6.22, "altitude": 400},
    "Vaucluse": {"lat": 44.00, "lon": 5.17, "altitude": 400},
    "Vendee": {"lat": 46.67, "lon": -1.33, "altitude": 50},
    "Vienne": {"lat": 46.58, "lon": 0.50, "altitude": 150},
    "Vosges": {"lat": 48.17, "lon": 6.42, "altitude": 500},
    "Yonne": {"lat": 47.83, "lon": 3.58, "altitude": 200},
    "Yvelines": {"lat": 48.83, "lon": 1.83, "altitude": 100},
}


def load_silver_data() -> dict[str, pd.DataFrame]:
    """Load all silver (cleaned) data files."""
    data = {}

    for file_path in SILVER_PATH.glob("*.parquet"):
        name = file_path.stem
        data[name] = pd.read_parquet(file_path)
        print(f"  [OK] Loaded {name}: {data[name].shape}")

    return data


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS - YEARLY AGGREGATES
# =============================================================================


def add_yearly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Add yearly statistics on climate metrics per department."""
    aggregated = (
        df.groupby(["nom_dep", "year"])
        .agg(
            # Temperature features
            temp_mean_year=("near_surface_air_temperature", "mean"),
            temp_max_year=("daily_maximum_near_surface_air_temperature", "max"),
            temp_min_year=("near_surface_air_temperature", "min"),
            temp_std_year=("near_surface_air_temperature", "std"),
            # Precipitation features
            precip_sum_year=("precipitation", "sum"),
            precip_std_year=("precipitation", "std"),
            precip_max_day=("precipitation", "max"),
        )
        .reset_index()
    )
    return aggregated


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS - SEASONAL AGGREGATES
# =============================================================================


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add seasonal (winter/spring/summer/autumn) climate features."""
    # Add month column
    df = df.copy()
    df["month"] = df["time"].dt.month

    seasonal_features = []

    for season_name, months in SEASONS.items():
        season_df = df[df["month"].isin(months)]

        season_agg = (
            season_df.groupby(["nom_dep", "year"])
            .agg(
                **{
                    f"temp_mean_{season_name}": ("near_surface_air_temperature", "mean"),
                    f"temp_max_{season_name}": (
                        "daily_maximum_near_surface_air_temperature",
                        "max",
                    ),
                    f"precip_sum_{season_name}": ("precipitation", "sum"),
                }
            )
            .reset_index()
        )
        seasonal_features.append(season_agg)

    # Merge all seasonal features
    result = seasonal_features[0]
    for sf in seasonal_features[1:]:
        result = result.merge(sf, on=["nom_dep", "year"], how="outer")

    return result


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS - EXTREME EVENTS
# =============================================================================


def add_extreme_temperature_days(df: pd.DataFrame) -> pd.DataFrame:
    """Add count of extreme temperature days (high heat and frost)."""
    # Calculate thresholds from entire dataset
    high_temp_threshold = df["near_surface_air_temperature"].quantile(0.95)
    low_temp_threshold = df["near_surface_air_temperature"].quantile(0.05)

    df = df.copy()
    df["is_high_heat_day"] = df["near_surface_air_temperature"] > high_temp_threshold
    df["is_frost_day"] = df["near_surface_air_temperature"] < low_temp_threshold

    extreme_counts = (
        df.groupby(["nom_dep", "year"])
        .agg(
            n_high_heat_days=("is_high_heat_day", "sum"),
            n_frost_days=("is_frost_day", "sum"),
        )
        .reset_index()
    )

    return extreme_counts


def add_spring_frost_days(df: pd.DataFrame) -> pd.DataFrame:
    """Add count of frost days specifically in spring (Mar-May) - critical for crops."""
    df = df.copy()
    df["month"] = df["time"].dt.month

    # Frost threshold (0°C in Kelvin)
    frost_threshold = 273.15

    spring_df = df[df["month"].isin([3, 4, 5])]
    spring_df = spring_df.copy()
    spring_df["is_spring_frost"] = spring_df["near_surface_air_temperature"] < frost_threshold

    spring_frost = (
        spring_df.groupby(["nom_dep", "year"])
        .agg(n_spring_frost_days=("is_spring_frost", "sum"))
        .reset_index()
    )

    return spring_frost


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS - HEATWAVES & DROUGHT
# =============================================================================


def add_heatwave_days(
    df: pd.DataFrame, threshold_kelvin: float = 303.0, min_consecutive: int = 3
) -> pd.DataFrame:
    """
    Count days under heatwave conditions per year.
    Heatwave = consecutive days with max temp > threshold.
    """
    df = df.copy()
    df = df.sort_values(["nom_dep", "time"]).reset_index(drop=True)

    df["is_hot"] = df["daily_maximum_near_surface_air_temperature"] > threshold_kelvin

    # Identify consecutive hot day streaks
    df["streak_id"] = (df["is_hot"] != df.groupby("nom_dep")["is_hot"].shift()).cumsum()

    # Count streak lengths
    streak_lengths = df[df["is_hot"]].groupby(["nom_dep", "streak_id"]).size()

    # Mark days that are part of heatwaves (streak >= min_consecutive)
    heatwave_streaks = streak_lengths[streak_lengths >= min_consecutive]
    heatwave_keys = set(heatwave_streaks.index)

    df["is_heatwave_day"] = df.apply(
        lambda row: (row["nom_dep"], row["streak_id"]) in heatwave_keys if row["is_hot"] else False,
        axis=1,
    )

    heatwave_counts = (
        df.groupby(["nom_dep", "year"])
        .agg(n_heatwave_days=("is_heatwave_day", "sum"))
        .reset_index()
    )

    return heatwave_counts


def add_drought_periods(
    df: pd.DataFrame, precip_quantile: float = 0.10, min_consecutive: int = 7
) -> pd.DataFrame:
    """
    Count days under drought conditions per year.
    Drought = consecutive days with very low precipitation.
    """
    df = df.copy()
    df = df.sort_values(["nom_dep", "time"]).reset_index(drop=True)

    # Fixed: use precipitation column, not temperature
    threshold = df["precipitation"].quantile(precip_quantile)
    df["is_dry"] = df["precipitation"] < threshold

    # Identify consecutive dry day streaks
    df["streak_id"] = (df["is_dry"] != df.groupby("nom_dep")["is_dry"].shift()).cumsum()

    # Count streak lengths
    streak_lengths = df[df["is_dry"]].groupby(["nom_dep", "streak_id"]).size()

    # Mark days that are part of droughts (streak >= min_consecutive)
    drought_streaks = streak_lengths[streak_lengths >= min_consecutive]
    drought_keys = set(drought_streaks.index)

    df["is_drought_day"] = df.apply(
        lambda row: (row["nom_dep"], row["streak_id"]) in drought_keys if row["is_dry"] else False,
        axis=1,
    )

    drought_counts = (
        df.groupby(["nom_dep", "year"]).agg(n_drought_days=("is_drought_day", "sum")).reset_index()
    )

    return drought_counts


def add_wet_periods(
    df: pd.DataFrame, precip_quantile: float = 0.90, min_consecutive: int = 3
) -> pd.DataFrame:
    """
    Count days under heavy rain conditions per year.
    Wet period = consecutive days with high precipitation.
    """
    df = df.copy()
    df = df.sort_values(["nom_dep", "time"]).reset_index(drop=True)

    # Fixed: use precipitation column, not temperature
    threshold = df["precipitation"].quantile(precip_quantile)
    df["is_wet"] = df["precipitation"] > threshold

    # Identify consecutive wet day streaks
    df["streak_id"] = (df["is_wet"] != df.groupby("nom_dep")["is_wet"].shift()).cumsum()

    # Count streak lengths
    streak_lengths = df[df["is_wet"]].groupby(["nom_dep", "streak_id"]).size()

    # Mark days that are part of wet periods (streak >= min_consecutive)
    wet_streaks = streak_lengths[streak_lengths >= min_consecutive]
    wet_keys = set(wet_streaks.index)

    df["is_wet_period_day"] = df.apply(
        lambda row: (row["nom_dep"], row["streak_id"]) in wet_keys if row["is_wet"] else False,
        axis=1,
    )

    wet_counts = (
        df.groupby(["nom_dep", "year"])
        .agg(n_wet_period_days=("is_wet_period_day", "sum"))
        .reset_index()
    )

    return wet_counts


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS - AGRONOMIC INDICES
# =============================================================================


def add_growing_degree_days(
    df: pd.DataFrame, base_temp_kelvin: float = GDD_BASE_TEMP_KELVIN
) -> pd.DataFrame:
    """
    Calculate Growing Degree Days (GDD) - accumulated heat units for crop development.
    GDD = sum of max(0, daily_mean_temp - base_temp) over growing season.
    """
    df = df.copy()

    # GDD for each day
    df["gdd"] = np.maximum(0, df["near_surface_air_temperature"] - base_temp_kelvin)

    # Also calculate for growing season only (Mar-Aug for barley)
    df["month"] = df["time"].dt.month
    df["gdd_growing_season"] = np.where(df["month"].isin([3, 4, 5, 6, 7, 8]), df["gdd"], 0)

    gdd_features = (
        df.groupby(["nom_dep", "year"])
        .agg(
            gdd_total_year=("gdd", "sum"),
            gdd_growing_season=("gdd_growing_season", "sum"),
        )
        .reset_index()
    )

    return gdd_features


def add_temperature_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average daily temperature range (diurnal variation).
    Large swings can stress crops.
    """
    df = df.copy()
    df["temp_range"] = (
        df["daily_maximum_near_surface_air_temperature"] - df["near_surface_air_temperature"]
    )

    temp_range_features = (
        df.groupby(["nom_dep", "year"])
        .agg(
            temp_range_mean=("temp_range", "mean"),
            temp_range_max=("temp_range", "max"),
        )
        .reset_index()
    )

    return temp_range_features


def add_geographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add geographic features: latitude, longitude, altitude.
    These capture regional differences in growing conditions.
    """
    df = df.copy()

    df["latitude"] = df["nom_dep"].map(lambda x: DEPARTMENT_GEO.get(x, {}).get("lat", np.nan))
    df["longitude"] = df["nom_dep"].map(lambda x: DEPARTMENT_GEO.get(x, {}).get("lon", np.nan))
    df["altitude"] = df["nom_dep"].map(lambda x: DEPARTMENT_GEO.get(x, {}).get("altitude", np.nan))

    # Check for missing departments
    missing = df[df["latitude"].isna()]["nom_dep"].unique()
    if len(missing) > 0:
        print(f"    [WARN] Missing geo data for: {missing}")

    return df


def add_trend_features(df: pd.DataFrame, base_year: int = 1982) -> pd.DataFrame:
    """
    Add trend features to capture temporal changes in yields.
    Yields have generally increased over time due to improved practices.
    """
    df = df.copy()

    # Linear trend (years since base year)
    df["year_trend"] = df["year"] - base_year

    # Quadratic trend (for non-linear changes)
    df["year_trend_sq"] = df["year_trend"] ** 2

    return df


# =============================================================================
# MAIN FEATURE ENGINEERING PIPELINE
# =============================================================================


def engineer_climate_features(df: pd.DataFrame, dataset_name: str = "") -> pd.DataFrame:
    """
    Apply all feature engineering transformations to climate data.
    Returns yearly aggregated features per department.
    """
    print(f"  Engineering features for {dataset_name}...")

    # 1. Yearly aggregates
    print("    -> Yearly aggregates...")
    features = add_yearly_aggregates(df)

    # 2. Seasonal features
    print("    -> Seasonal features...")
    seasonal = add_seasonal_features(df)
    features = features.merge(seasonal, on=["nom_dep", "year"], how="left")

    # 3. Extreme temperature days
    print("    -> Extreme temperature days...")
    extreme_temps = add_extreme_temperature_days(df)
    features = features.merge(extreme_temps, on=["nom_dep", "year"], how="left")

    # 4. Spring frost days
    print("    -> Spring frost days...")
    spring_frost = add_spring_frost_days(df)
    features = features.merge(spring_frost, on=["nom_dep", "year"], how="left")

    # 5. Heatwave days
    print("    -> Heatwave days...")
    heatwave = add_heatwave_days(df)
    features = features.merge(heatwave, on=["nom_dep", "year"], how="left")

    # 6. Drought periods
    print("    -> Drought periods...")
    drought = add_drought_periods(df)
    features = features.merge(drought, on=["nom_dep", "year"], how="left")

    # 7. Wet periods
    print("    -> Wet periods...")
    wet = add_wet_periods(df)
    features = features.merge(wet, on=["nom_dep", "year"], how="left")

    # 8. Growing Degree Days
    print("    -> Growing Degree Days...")
    gdd = add_growing_degree_days(df)
    features = features.merge(gdd, on=["nom_dep", "year"], how="left")

    # 9. Temperature range
    print("    -> Temperature range...")
    temp_range = add_temperature_range(df)
    features = features.merge(temp_range, on=["nom_dep", "year"], how="left")

    # 10. Geographic features
    print("    -> Geographic features (lat, lon, altitude)...")
    features = add_geographic_features(features)

    # 11. Trend features
    print("    -> Trend features (year_trend)...")
    features = add_trend_features(features)

    print(f"    [OK] {dataset_name}: {features.shape[0]} rows, {features.shape[1]} features")

    return features


def add_lagged_features(df: pd.DataFrame, lag_years: int = 1) -> pd.DataFrame:
    """Add previous year features (lagged features)."""
    df = df.copy()
    df = df.sort_values(["nom_dep", "year"]).reset_index(drop=True)

    # Columns to lag (excluding identifiers)
    cols_to_lag = [
        "precip_sum_year",
        "temp_mean_year",
        "gdd_growing_season",
        "n_drought_days",
    ]

    for col in cols_to_lag:
        if col in df.columns:
            df[f"{col}_lag{lag_years}"] = df.groupby("nom_dep")[col].shift(lag_years)

    return df


def save_gold_data(df: pd.DataFrame, name: str) -> None:
    """Save feature-engineered data to gold layer."""
    GOLD_PATH.mkdir(parents=True, exist_ok=True)

    output_path = GOLD_PATH / f"{name}.parquet"
    df.to_parquet(output_path, index=False)
    print(f"  [OK] Saved {name}.parquet ({len(df):,} rows, {len(df.columns)} cols)")


def run_silver_to_gold() -> dict[str, pd.DataFrame]:
    """Main pipeline: Silver -> Gold."""
    print("=" * 60)
    print("SILVER -> GOLD PIPELINE")
    print("=" * 60)

    # Load silver data
    print("\n[1/5] Loading silver data...")
    silver_data = load_silver_data()

    gold_data = {}

    # Process historical climate + yield for training
    print("\n[2/5] Processing TRAINING data (historical + yield)...")
    print(f"  -> Using years {TRAIN_YEAR_START}-{TRAIN_YEAR_END} only")

    climate_hist = silver_data["climate_historical"]
    yield_data = silver_data["barley_yield"]

    # Filter yield to training years
    yield_train = yield_data[
        (yield_data["year"] >= TRAIN_YEAR_START) & (yield_data["year"] <= TRAIN_YEAR_END)
    ].copy()
    print(f"  -> Yield data filtered: {len(yield_data)} -> {len(yield_train)} rows")

    # Engineer features on historical climate
    climate_features = engineer_climate_features(climate_hist, "climate_historical")

    # Add lagged features
    print("  -> Adding lagged features...")
    climate_features = add_lagged_features(climate_features)

    # Merge with yield data (yield as target, area/production in silver for later)
    print("  -> Merging with yield data...")
    training_data = climate_features.merge(
        yield_train[["nom_dep", "year", "yield"]],
        on=["nom_dep", "year"],
        how="inner",
    )
    print(f"  [OK] Training data: {training_data.shape}")

    gold_data["training_data"] = training_data

    # Process scenario datasets for predictions
    print("\n[3/5] Processing SCENARIO datasets for predictions...")

    for scenario in ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]:
        key = f"climate_{scenario}"
        if key in silver_data:
            scenario_features = engineer_climate_features(silver_data[key], key)
            # Add lagged features (will have NaN for first year)
            scenario_features = add_lagged_features(scenario_features)
            gold_data[f"{scenario}_features"] = scenario_features

    # Validate
    print("\n[4/5] Validating gold data...")
    for name, df in gold_data.items():
        null_counts = df.isnull().sum().sum()
        print(f"  -> {name}: {df.shape}, {null_counts} null values")

    # Save
    print("\n[5/5] Saving gold data...")
    for name, df in gold_data.items():
        save_gold_data(df, name)

    print("\n" + "=" * 60)
    print("SILVER -> GOLD COMPLETE")
    print("=" * 60)
    print(f"\nOutput files in: {GOLD_PATH}")
    print("  - training_data.parquet (for model training/testing)")
    print("  - ssp1_2_6_features.parquet (scenario 1 predictions)")
    print("  - ssp2_4_5_features.parquet (scenario 2 predictions)")
    print("  - ssp5_8_5_features.parquet (scenario 3 predictions)")

    # Print feature summary
    print(f"\n Features created ({len(training_data.columns)} total):")
    feature_cols = [c for c in training_data.columns if c not in ["nom_dep", "year", "yield"]]
    for col in feature_cols:
        print(f"  - {col}")

    return gold_data


if __name__ == "__main__":
    run_silver_to_gold()
