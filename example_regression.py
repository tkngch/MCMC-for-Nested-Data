#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import numpy
import scipy.stats
from mcmc_hierarchical_bayes import run_mcmc
numpy.random.seed(12345)


def generate_data(n_groups, n_responses_per_group):
    """
    This generates mock data to fit a regression model.
    """

    n = n_groups * n_responses_per_group

    # generate predictors
    x = numpy.hstack([
        numpy.tile([1], (n))[numpy.newaxis].T,
        numpy.random.normal(size=(n, 1))
    ])

    # generate true parameter values
    beta = numpy.hstack([
        numpy.repeat(  # intercept
            numpy.random.normal(loc=0, scale=1, size=n_groups),
            [n_responses_per_group] * n_groups)[numpy.newaxis].T,
        numpy.repeat(  # slope
            numpy.random.normal(loc=100, scale=100, size=n_groups),
            [n_responses_per_group] * n_groups)[numpy.newaxis].T
    ])

    # generate response variable with noise
    y = numpy.sum(x * beta, axis=1) + numpy.random.normal(size=n)

    # indicate which row belongs to which group
    group = numpy.repeat(range(n_groups), [n_responses_per_group] * n_groups)

    return {"group": group, "X": x, "y": y}


def compute_loglikelihood(parameter, data):
    """

    Computes log likelihood with the regression model. This function assumes
    that each group has three parameters: intercept, slope, and noise sd.

    """

    # re-organise parameter values
    beta_hat = numpy.hstack([
        numpy.array([[parameter[0][i]] for i in data["group"]]),
        numpy.array([[parameter[1][i]] for i in data["group"]])
    ])

    y_hat = numpy.sum(data["X"] * beta_hat, axis=1)
    noise = numpy.array([parameter[2][i] for i in data["group"]])

    ll = scipy.stats.norm(loc=data["y"], scale=noise).logpdf(y_hat)
    return ll


def main():
    # mcmc configuration
    n_chains = 4
    n_iter = 2000
    n_samples = 100

    outputdir = "./sample/example_regression/"

    # paramter specification
    parameter_name = ("b0", "b1", "sigma")
    parameter_family = {"b0": "gaussian",
                        "b1": "gaussian",
                        "sigma": "exponential"}
    parameter_value_range = {"b0": [-100, 100],
                             "b1": [-100, 100],
                             "sigma": [0.00, 100.]}

    # generate data and partial apply to the loglikelihood function
    n_groups = 10
    n_responses_per_group = 10
    data = generate_data(n_groups, n_responses_per_group)
    objective_func = functools.partial(compute_loglikelihood, data=data)

    # run mcmc and make figures
    run_mcmc(parameter_name, parameter_family, parameter_value_range,
             n_groups, n_responses_per_group,
             objective_func,
             n_chains, n_iter, n_samples, outputdir,
             start_with_mle=False)


if __name__ == "__main__":
    main()
