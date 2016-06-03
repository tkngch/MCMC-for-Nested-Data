#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functools

import numpy
import scipy.stats

from posteriorSampling import samplePosterior
from sampleDiagnosis import diagnoseSamples

numpy.random.seed(12345)


def generateData(nGroups, nResponsesPerGroup):
    """
    This generates mock data to fit a regression model.
    """

    n = nGroups * nResponsesPerGroup

    # generate predictors
    x = numpy.hstack([
        numpy.tile([1], (n))[numpy.newaxis].T,
        numpy.random.normal(size=(n, 1))
    ])

    # generate true parameter values
    beta = numpy.hstack([
        numpy.repeat(  # intercept
            numpy.random.normal(loc=0, scale=1, size=nGroups),
            [nResponsesPerGroup] * nGroups)[numpy.newaxis].T,
        numpy.repeat(  # slope
            numpy.random.normal(loc=100, scale=100, size=nGroups),
            [nResponsesPerGroup] * nGroups)[numpy.newaxis].T
    ])

    # generate response variable with noise
    y = numpy.sum(x * beta, axis=1) + numpy.random.normal(size=n)

    trueValueString = "\nTrue value:\n"
    for i in range(beta.shape[1]):
        trueValueString += "\tbeta%i: {mean: %.2f, sd: %.2f}\n"\
              % (i, numpy.mean(beta[:, i]), numpy.std(beta[:, 1]))

    # indicate which row belongs to which group
    group = numpy.repeat(range(nGroups), [nResponsesPerGroup] * nGroups)

    return {"group": group, "X": x, "y": y}, trueValueString


def computeLogLikelihood(parameter, data):
    """

    Computes log likelihood with the regression model. This function assumes
    that each group has three parameters: intercept, slope, and noise sd.

    """

    betaHat = numpy.vstack([parameter[0], parameter[1]]).T

    yHat = numpy.sum(data["X"] * betaHat, axis=1)
    noise = numpy.array(parameter[2])

    ll = scipy.stats.norm(loc=data["y"], scale=noise).logpdf(yHat)
    return ll


def main(pooling):
    # mcmc configuration
    nChains = 4
    nIter = 2000
    nSamples = 1000

    outputDirectory = "./example/sample/regression/"

    # paramter specification
    parameterName = ("b0", "b1", "sigma")
    startingPointValueRange = {"b0": [-100, 100],
                               "b1": [0, 200],
                               "sigma": [0.00, 100.]}
    prior = [scipy.stats.norm(loc=0, scale=10),
             scipy.stats.norm(loc=100, scale=10),
             scipy.stats.gamma(10)]

    # generate data and partial apply to the loglikelihood function
    nGroups = 10
    nResponsesPerGroup = 10
    data, trueValueString = generateData(nGroups, nResponsesPerGroup)
    objectiveFunction = functools.partial(computeLogLikelihood, data=data)

    # run mcmc
    samplePosterior(nChains, nIter, nSamples,
                    parameterName, nGroups, nResponsesPerGroup,
                    pooling, objectiveFunction,
                    outputDirectory,
                    priorDistribution=prior,
                    startWithMLE=True,
                    startingPointValueRange=startingPointValueRange,
                    nProcesses=0)

    print(trueValueString)
    diagnoseSamples(outputDirectory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example MCMC for a linear regression.")

    parser.add_argument(
        "pooling", nargs="?", default="partial",
        help="Pooling method (optional) : partial, complete or none. " +
        "Default is partial."
    )

    args = parser.parse_args()

    main(args.pooling)
