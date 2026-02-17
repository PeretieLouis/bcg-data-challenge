# ASSUMING DATA IS ALREADY SORTED AND CLEANED
# Features are computed yearly & per departement.

############################# Yearly informations (aggregates) ###########################
def adding_yearly_aggregates_per_department(dataset):
    """Adding yearly statistics on metrics per departement."""
    dataset_added_feature = dataset.copy()
    aggregated_data = (
        dataset_added_feature.groupby(["nom_dep", "year"])
        .agg(
            mean_near_surface_air_temperature_of_the_year_per_department=(
                "near_surface_air_temperature",
                "mean",
            ),
            max_daily_near_surface_air_temperature_of_the_year_per_department=(
                "daily_maximum_near_surface_air_temperature",
                "max",
            ),
            sum_precipitation_of_the_year_per_department=("precipitation", "sum"),
            variability_of_temperature_of_the_year_per_department=(
                "near_surface_air_temperature",
                "std",
            ),
            variability_of_precipitation_of_the_year_per_department=("precipitation", "std"),
        )
        .reset_index()
    )
    return dataset_added_feature.merge(aggregated_data, on=["nom_dep", "year"])


############################# Temperature informations ###########################
def add_number_of_days_with_temp_exceeding_quantile(dataset, quantile=0.95):
    """Adding the total numbers in the year per departement that exceeds a certain
    threshold (quantile) of temperature (quantile of the whole country)."""
    dataset_added_feature = dataset.copy()
    threshold = dataset_added_feature["near_surface_air_temperature"].quantile(quantile)
    result = (
        dataset_added_feature.groupby(["year", "nom_dep"])["near_surface_air_temperature"]
        .apply(lambda x: (x > threshold).sum())
        .rename("yearly_number_of_high_heat_days_per_departement")
        .reset_index()
    )

    return dataset_added_feature.merge(result, on=["nom_dep", "year"])


def add_number_of_days_below_quantile(dataset, quantile=0.05):
    """Adding the total numbers in the year that are below a certain
    threshold (quantile) of temperature."""
    dataset_added_feature = dataset.copy()
    threshold = dataset_added_feature["near_surface_air_temperature"].quantile(quantile)
    result = (
        dataset_added_feature.groupby(["year", "nom_dep"])["near_surface_air_temperature"]
        .apply(lambda x: (x < threshold).sum())
        .rename("yearly_number_of_frost_days_per_departement")
        .reset_index()
    )

    return dataset_added_feature.merge(result, on=["nom_dep", "year"])


############################# Heatwaves informations ###########################
def compute_yearly_nb_days_under_heatwaves_per_departement(dataset, threshold=303, min_length=3):
    """Computing the total number of days that are under a period of heatwave per departement
    (i.e. number of days hotter than a certain threshold AND within a period of
    successive days (streak) of at least a minimum length)"""
    dataset_added_feature = dataset.copy()
    hot = dataset_added_feature["daily_maximum_near_surface_air_temperature"] > threshold
    groups = (
        hot.groupby(dataset_added_feature["nom_dep"])
        .apply(lambda x: (x != x.shift()).cumsum())
        .reset_index(level=0, drop=True)
    )

    streaks = hot.groupby([dataset_added_feature["nom_dep"], groups]).transform("sum")
    heatwave_days = (hot & (streaks >= min_length)).astype(int)

    dataset_added_feature["is_heatwave_day"] = heatwave_days

    return dataset_added_feature.merge(
        dataset_added_feature.groupby(["year", "nom_dep"])["is_heatwave_day"]
        .sum()
        .rename("nb_days_under_heatwave")
        .reset_index(),
        on=["year", "nom_dep"],
    )


def compute_yearly_nb_consecutive_dry_days_per_departement(dataset, quantile=0.07, min_length=3):
    """Computing the total number of consecutive days that are considered dry per departement
    (i.e. number of days drier than a certain threshold AND within a period of
    successive days (streak) of at least a minimum length)"""
    dataset_added_feature = dataset.copy()
    threshold = dataset_added_feature["precipitation"].quantile(quantile)
    dry = dataset_added_feature["daily_maximum_near_surface_air_temperature"] < threshold
    groups = (
        dry.groupby(dataset_added_feature["nom_dep"])
        .apply(lambda x: (x != x.shift()).cumsum())
        .reset_index(level=0, drop=True)
    )

    streaks = dry.groupby([dataset_added_feature["nom_dep"], groups]).transform("sum")

    dry_days = (dry & (streaks >= min_length)).astype(int)

    dataset_added_feature["is_dry_day"] = dry_days

    return dataset_added_feature.merge(
        dataset_added_feature.groupby(["year", "nom_dep"])["is_dry_day"]
        .sum()
        .rename("nb_days_dry")
        .reset_index(),
        on=["year", "nom_dep"],
    )


def compute_yearly_nb_consecutive_wet_days_per_departement(dataset, quantile=0.93, min_length=3):
    """Computing the total number of consecutive days that under a period of heavy
    precipitation per departement
    (i.e. number of days wetter than a certain threshold AND within a period of
    successive days (streak) of at least a minimum length)"""
    dataset_added_feature = dataset.copy()
    threshold = dataset_added_feature["precipitation"].quantile(quantile)
    wet = dataset_added_feature["daily_maximum_near_surface_air_temperature"] > threshold
    groups = (
        wet.groupby(dataset_added_feature["nom_dep"])
        .apply(lambda x: (x != x.shift()).cumsum())
        .reset_index(level=0, drop=True)
    )

    streaks = wet.groupby([dataset_added_feature["nom_dep"], groups]).transform("sum")
    wet_days = (wet & (streaks >= min_length)).astype(int)

    dataset_added_feature["is_wet_day"] = wet_days

    return dataset_added_feature.merge(
        dataset_added_feature.groupby(["year", "nom_dep"])["is_wet_day"]
        .sum()
        .rename("nb_days_wet")
        .reset_index(),
        on=["year", "nom_dep"],
    )


############################# Precipitation informations ###########################
def adding_nb_dry_days(dataset, quantile=0.1):
    """Adding number of dry days, according to precipitation ammounts per departement ."""
    dataset_added_feature = dataset.copy()
    threshold = dataset_added_feature["precipitation"].quantile(quantile)

    result = (
        dataset_added_feature.groupby(["year", "nom_dep"])["precipitation"]
        .apply(lambda x: (x < threshold).sum())
        .rename("yearly_number_of_dry_days_per_departement")
        .reset_index()
    )

    return dataset_added_feature.merge(result, on=["nom_dep", "year"])


def adding_nb_wet_days(dataset, quantile=0.9):
    """Adding number of wet days, according to precipitation ammounts per departement ."""
    dataset_added_feature = dataset.copy()
    threshold = dataset_added_feature["precipitation"].quantile(quantile)

    result = (
        dataset_added_feature.groupby(["year", "nom_dep"])["precipitation"]
        .apply(lambda x: (x > threshold).sum())
        .rename("yearly_number_of_wet_days_per_departement")
        .reset_index()
    )

    return dataset_added_feature.merge(result, on=["nom_dep", "year"])
