# loading packages
import os

# linear algebra  and simulation stack
import numpy as np
from numba import njit
from scipy.integrate import odeint


# data wrangling stack
import polars as pl
import plotnine as p9
import datetime

# paramter eatimation stack
import pymc as pm

# import arviz as az
import preliz as pz
from scipy import stats as stats

p9.theme_set(p9.theme_bw(base_size=13))
# start with something basic ---
# make some pretty distribution plots by generating from distributions

np.random.seed(72)


def gen_random_poisson_distributions(mean, size):
    """
    for a given "mean" draw a possion random sample of size "size"
    """
    return pz.Poisson(mean).rvs(size)


# mean_vals to simulate
mean_vals = [1, 5, 9, 15, 20]
sample_size = 1000

simulation_list = []
for mean in mean_vals:
    sim = pl.DataFrame(
        {
            "mean_val": [mean] * sample_size,
            "sample": gen_random_poisson_distributions(mean=mean, size=sample_size),
        }
    )
    simulation_list.append(sim)

# make on data for plotting
simulation_df = pl.concat(simulation_list)

simulation_plt = (
    p9.ggplot(
        simulation_df.with_columns(
            mean_val_c=pl.col("mean_val")
            .cast(pl.Utf8)
            .cast(pl.Enum([str(m) for m in mean_vals]))
        )
    )
    + p9.geom_histogram(
        p9.aes(x="sample", fill="mean_val_c"), position="identity", alpha=0.7, bins=23
    )
    + p9.geom_vline(
        p9.aes(xintercept="mean_val", color="mean_val_c"), linetype="dashed", size=0.8
    )
    + p9.labs(x="Sample", y="Frequency", fill="Mean\nValue", colour="Mean\nValue")
)

simulation_plt


# lets see if I can work with differential equations
# a simple disease model wihth demography --
@njit
def seir_log_skel(logX, t, theta):
    """
    Log-transformed SEIR model with demography, seasonality, and cumulative incidence.

    Parameters:
    logX: [logS, logE, logI, logR, C]
    t: time (in years)
    theta: [mu, R0, beta1, sigma, gamma]

    Returns:
    Derivatives of [logS, logE, logI, logR, C]
    """
    x, e, y, r, C = logX
    mu, R0, beta1, sigma, gamma = theta[:5]

    # Reasonable bounds
    minval = 1e-12
    maxval = 1e9  # Adjust as needed for your scale

    # Reconstruct compartments with clamping
    S = min(max(np.exp(x), minval), maxval)
    E = min(max(np.exp(e), minval), maxval)
    I = min(max(np.exp(y), minval), maxval)
    R = min(max(np.exp(r), minval), maxval)
    N = S + E + I + R

    # Time-varying transmission rate
    beta0 = R0 * gamma
    beta_t = beta0 * (1 + beta1 * np.sin(2 * np.pi * t))
    lambda_t = beta_t * I / max(N, 1e-12)

    # Derivatives
    dx = mu * N * np.exp(-x) - lambda_t - mu
    de = lambda_t * np.exp(x - e) - sigma - mu
    dy = sigma * np.exp(e - y) - gamma - mu
    dr = gamma * np.exp(y - r) - mu
    dC = gamma * I  # raw value, not log-transformed

    return np.array([dx, de, dy, dr, dC])


# the


def generate_trajectory(
    S_0,
    E_0,
    I_0,
    R_0,
    C_0,
    R0,
    mu,
    beta1,
    sigma,
    gamma,
    dt=1,
    nt_pts=20,
    etol=1e-12,
    long_form=True,
):
    """
    Runs the differential equation solver and collects the solution
    for a set of paramter values

    Inputs:
    ### Initial value parameters ###
    S_0: Initial number of susceptible individuals
    E_0: Initial number of exposed individuals
    I_0: Initial number of infected individuals
    R_0: Initial number of recovered individuals
    C_0: Initial number of cumulative cases

    ### Regular parameters ###
    R0: Basic reproduction number
    mu: Per capita death rate
    beta1: Amplitude of seasonal forcing in the transmission rate
    sigma: Rate of progression from exposed to infected
    gamma: Rate of recovery

    ### Simulation parameters ###
    dt: Time step for the simulation
    nt_pts: Number of time points to simulate

    Returns a polars DataFrame with the columns:
    """

    # generate a time array
    time = np.arange(0, nt_pts, dt)

    # log-transform initial conditions (except C)
    logX0 = np.log(np.array([S_0, E_0, I_0, R_0, C_0]) + etol)  # avoid log(0) crash

    # pack all the variables
    theta = np.array([mu, R0, beta1, sigma, gamma])
    # solve the differential equations
    sol = odeint(
        func=seir_log_skel, y0=logX0, t=time, args=(theta,), rtol=1e-6, atol=1e-9
    )

    # extract and transform
    sol_linear = np.empty_like(sol)
    sol_linear[:, :4] = np.exp(sol[:, :4])  # S, E, I, R
    sol_linear[:, 4] = sol[:, 4]  # C (linear)

    result = pl.DataFrame(sol_linear, schema=["S", "E", "I", "R", "C"]).with_columns(
        C=(pl.col("C") - pl.col("C").shift(1)).fill_null(0), time=time
    )

    if long_form == True:
        result = result.unpivot(
            on=["S", "E", "I", "R", "C"],
            index=["time"],
            variable_name="compartment",
            value_name="size",
        ).with_columns(
            pl.col("compartment").cast(
                pl.Enum([str(c) for c in ["S", "E", "I", "R", "C"]])
            )
        )

    return result


# testing the differential equations  and making a simple plot
plot_data = generate_trajectory(
    S_0=10000,
    E_0=0,
    I_0=10,
    R_0=(100000 - 10 - 10000),
    C_0=0,
    mu=1 / 70,
    R0=2,
    beta1=0.6,
    sigma=1 / (3 / 365.25),
    gamma=1 / (4 / 365.25),
    dt=1 / 100,
    nt_pts=200,
)

traj_plot = (
    p9.ggplot(
        plot_data.filter(pl.col("time") > 180),
        p9.aes(x="time", y="size", color="compartment"),
    )
    + p9.geom_line()
    + p9.facet_wrap("compartment", scales="free_y")
    + p9.labs(x="Time (years)", y="Size", color="Compartment")
    + p9.theme(legend_position="bottom")
)

traj_plot
