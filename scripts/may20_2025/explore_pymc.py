from sim_utils import sim_seir as ss

import numpy as np
import polars as pl
import plotnine as p9

import pymc as pm
import arviz as az

import pytensor.tensor as pt
from pytensor.compile.ops import as_op
from scipy.integrate import odeint
from scipy.optimize import least_squares

from scipy import stats as stats

p9.theme_set(p9.theme_bw())


data = ss.plot_data.filter(
    (pl.col("compartment") == "Obs") & (pl.col("time") > 0)
).drop("compartment")


data_plt = (
    p9.ggplot(data, p9.aes(x="time", y="size"))
    + p9.geom_point(shape="o", fill="white", colour="red", size=2.5)
    + p9.labs(x="Time (years)", y="Cases")
)
data_plt
# data looks great!


def obj_func(
    params: np.ndarray, data: pl.dataframe, model_func: callable
) -> np.ndarray:

    t = data.filter()

    return


obj_func(params=[3, 0.2], data=ss.plot_data, model_func=ss.seir_log_skel)
