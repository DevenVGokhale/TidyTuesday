# loading packages
import os
#import tqdm as tqdm
import polars as pl
import plotnine as p9
import datetime

# import pymc as pm
# import arviz as az
# import preliz as pz

# this is where the data is 
import pydytuesday
pydytuesday.get_date('2025-05-20')

# read data from csv
water_df = pl.read_csv("water_quality.csv", null_values=["", "NA", "N/A"]) 
weather_df = pl.read_csv("weather.csv")

# look at the datasets 
print(water_df.schema)
print(weather_df.schema)

# transform the data
# lis the columns that are integers rihght now
cols_to_cast = ["enterococci_cfu_100ml", "water_temperature_c", "conductivity_ms_cm"]

# transform the water data
water_tdf = (
    water_df
    .with_columns(
        date_time=pl.concat_str(["date", "time"], separator=" ")
    )
    .with_columns(
        pl.col("date_time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"),
        pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"),
        pl.col("time").str.strptime(pl.Time, format="%H:%M:%S"), 
        water_temperature_corrected=
        pl.when(pl.col("water_temperature_c") > 60)
        .then(None)
        .otherwise(pl.col("water_temperature_c"))
    )
    .with_columns(
        [pl.col(c).cast(pl.Float64) for c in cols_to_cast]
    )
    .drop("water_temperature_c")
)

#transform the weather data
weather_tdf = (
    weather_df
    .with_columns(
        pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"), 
        )
    .drop("latitude", "longitude")    
    )

print(water_tdf.schema)
print(weather_tdf.schema)


# join the two datasets
joined_df = (
    water_tdf
    .join(weather_tdf, on=["date"], how="inner")
    .with_columns(
        latitude_c=pl.col("latitude").cast(pl.Utf8),
        longitude_c=pl.col("longitude").cast(pl.Utf8)
    )
    .sort(["swim_site", "date"])       
    )

# now working with the joined data 
on_vars = ["log_bact_load_1", "water_temperature_corrected", "log_conductivity_1", 
           "max_temp_C", "min_temp_C", "precipitation_mm"]

joined_long = (
    joined_df
    .with_columns(
        log_bact_load_1=(pl.col("enterococci_cfu_100ml") + 1).log10(), 
        log_conductivity_1=(pl.col("conductivity_ms_cm") + 1).log10() 
    )  
    .drop("enterococci_cfu_100ml", "conductivity_ms_cm") 
    .unpivot(
        index=["date", "swim_site"], 
        on=on_vars, 
        variable_name="covariates", 
        value_name="value"
    )
    .filter(pl.col("date") > datetime.date(2010,1,1))
    .group_by("date", "covariates")
    .agg(
        value_low=pl.col("value").quantile(0.025, "nearest"),
        value_med=pl.col("value").quantile(0.5, "nearest"),
        value_high=pl.col("value").quantile(0.975, "nearest")
    )
)


joined_plt = (
    p9.ggplot(joined_long, p9.aes(x="date"))
    + p9.geom_line(p9.aes(y="value_med"))
    #+ p9.geom_ribbon(p9.aes(ymin="value_low", ymax="value_high"), alpha=0.9)
    + p9.facet_wrap("~covariates", scales="free")
    + p9.theme_bw()
)


# need to remove seasonality to from the data 
joined_df2 = (
    joined_df
    .with_columns(
        log_bact_load_1=(pl.col("enterococci_cfu_100ml") + 1).log10(), 
        log_conductivity_1=(pl.col("conductivity_ms_cm") + 1).log10() 
    )
    .filter(pl.col("date") >= datetime.date(2010,1,1))
)

# variables to detrend 
vars_to_detrend = ["log_bact_load_1", "log_conductivity_1", "max_temp_C", 
                   "min_temp_C", "precipitation_mm", "water_temperature_corrected"]

window_size_val = 52
exprs = (
    [(pl.col(v).rolling_mean(window_size=window_size_val))
            .over("swim_site")
            .alias(f"{v}_smooth")
            for v in vars_to_detrend
        ] +
    [(pl.col(v) - pl.col(v).rolling_mean(window_size=window_size_val))
            .over("swim_site")
            .alias(f"{v}_resid")
            for v in vars_to_detrend
        ]
    )

joined_df_detrended = (
    joined_df2
    .with_columns(exprs)
)

def gen_scatter_plot(var1, var2):
    plt = (
        p9.ggplot(joined_df_detrended, p9.aes(x=var1, y=var2, color="swim_site"))
        + p9.geom_point(size=3, alpha=0.5)
        + p9.theme_bw()
        + p9.theme(legend_position="none")
    )
    return plt

def gen_time_series_plot(var):
    plt = (
        p9.ggplot(joined_df_detrended, p9.aes(x="date", y=var, color="swim_site"))
        + p9.geom_line()
        + p9.theme_bw()
        + p9.theme(legend_position="none")
    )
    return plt

# make a grid plot 
c11 = gen_scatter_plot("precipitation_mm_resid", "log_bact_load_1_resid")
c21 = gen_scatter_plot("log_conductivity_1_resid", "log_bact_load_1_resid")
c12 = gen_scatter_plot("water_temperature_corrected_resid", "log_bact_load_1_resid")
c22 = gen_scatter_plot("max_temp_C_resid", "log_bact_load_1_resid")







