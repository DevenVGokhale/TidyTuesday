from sim_utils import sim_seir as ss

import numpy as np
import polars as pl
import plotnine as p9

import pymc as pm
import arviz as az

import pytensor.tensor as pt
from pytensor.compile.ops import as_op
from scipy.integrate import odeint
from scipy.special import logit, expit
from scipy.optimize import least_squares

from scipy import stats as stats
from scipy.stats import poisson

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


# dictionaries for the
params_to_est = np.array([np.log(2), logit(0.2)])

to_transform = {"R0": "log", "beta1": "logit"}

from_transform = {"R0": "exp", "beta1": "invlogit"}

params_val = {
    "mu": 1 / 70,
    "R0": 2.0,
    "beta1": 0.2,
    "sigma": 1 / (3 / 365.25),  # 3 days
    "gamma": 1 / (4 / 365.25),  # 4 days
    "S_0": ss.latest_per_compartment[0],  # initial susceptible population
    "E_0": ss.latest_per_compartment[1],  # initial exposed population
    "I_0": ss.latest_per_compartment[2],  # initial infected population
    "R_0": ss.latest_per_compartment[3],  # initial recovered population
    "C_0": ss.latest_per_compartment[4],  # initial cumulative cases
    "rho": 0.05,
}

eparams_n = ["R0", "beta1"]
iparams_n = ["S_0", "E_0", "I_0", "R_0", "C_0"]
rparams_n = ["mu", "R0", "beta1", "sigma", "gamma", "rho"]


def obj_func(
    guess: list,
    eparams: list,
    iparams: list,
    rparams: list,
    params: dict,
    data: pl.DataFrame,
    model_func: callable,
) -> np.ndarray:
    """
    For a given set of paramtere values and the obsrvation data, this function
    computes the sum of squared errors between the model predictions and the
    observed data. This function is used to optimize the parameters of the model
    using least squares optimization.
    Args:
        eparams (dict): dictionary of parameters to be estimated
        iparams (dict): dictionary of initial parameters
        rparams (dict): dictionary of rate parameters
        params (dict): dictionary of all parameters to be estimated
        data (pl.DataFrame): polars dataframe containing the observed data
        model_func (callable): function that defines the model to be used for simulation
    Returns:
        np.ndarray: returen a single value of the sum of squared errors
    """

    # unpack observations and time from the input data
    t = data.select("time").to_numpy().flatten()
    obs = data.select("size").to_numpy().flatten()

    # back transfrom the parameters to be fed into the model
    guess[0] = np.exp(guess[0])  # R0
    guess[1] = expit(guess[1])  # beta1

    # updated parameter dictionary with the parameters to be estimated guess
    if eparams is not None:
        updated_params = dict(zip(eparams, guess))
        params.update(updated_params)

    logX0 = np.log(
        np.array([params[k] for k in iparams]) + 1e-12
    )  # value to avoid log(0) crash
    theta = np.array([params[k] for k in rparams])

    sol = odeint(func=model_func, y0=logX0, t=t, args=(theta,), rtol=1e-6, atol=1e-9)
    # extract and transform back to the natural scale
    sol_linear = np.empty_like(sol)
    sol_linear[:, :4] = np.exp(sol[:, :4])  # S, E, I, R
    sol_linear[:, 4] = sol[:, 4]  # C (linear)

    # redefine the results as datfarame - doing this only for the polars convenience functions
    result = pl.DataFrame(sol_linear, schema=["S", "E", "I", "R", "C"]).with_columns(
        C=(pl.col("C") - pl.col("C").shift(1)).fill_null(0).clip(lower_bound=0),
        sim_obs=pl.col("C") * params["rho"],
    )

    sim_obs = result.select("sim_obs").to_numpy().flatten()

    # compute the sum of squared errors between the simulated observations and the actual observations
    score = np.sum((obs - sim_obs) ** 2)

    return score


obj_func(
    guess=params_to_est,
    eparams=eparams_n,
    iparams=iparams_n,
    rparams=rparams_n,
    params=params_val,
    data=data,
    model_func=ss.seir_log_skel,
)
