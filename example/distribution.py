#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import functools

import numpy
import scipy.stats

from posteriorSampling import samplePosterior
from sampleDiagnosis import diagnoseSamples

numpy.random.seed(12345)


def getFunction(parameterName, nGroups, nResponsesPerGroup):

    def _computeLogLikelihood(parameter, groupIndex, distributions):
        ll = numpy.zeros(len(groupIndex))
        for i, group in enumerate(groupIndex):
            ll[i] = sum([dist[group].logpdf(parameter[j][i])
                         for j, dist in enumerate(distributions)])

        return ll

    trueValueString = "\nTrue value:\n"

    distributions = []
    for i, name in enumerate(parameterName):
        mu = numpy.random.normal(loc=0, scale=1, size=nGroups)
        sd = numpy.random.gamma(1)

        tmp = [scipy.stats.norm(loc=loc, scale=sd) for loc in mu]
        distributions.append(tmp)

        trueValueString += "\t%s: {mean: %.2f, var: %.2f}\n" %\
            (name, numpy.mean(mu), numpy.var(mu))

    groupIndex = [i for i in range(nGroups) for j in range(nResponsesPerGroup)]

    func = functools.partial(_computeLogLikelihood,
                             groupIndex=groupIndex,
                             distributions=distributions)

    prior = [scipy.stats.norm(loc=0, scale=1) for name in parameterName]

    return func, prior, trueValueString


def main(pooling):
    nChains = 2
    nIter = 1000
    nSamples = 100

    outputDirectory = "./example/sample/distribution/"

    parameterName = ("a", "b", "c")

    nGroups = 10
    nResponsesPerGroup = 10

    objectiveFunction, prior, trueValueString =\
        getFunction(parameterName, nGroups, nResponsesPerGroup)

    samplePosterior(nChains, nIter, nSamples,
                    parameterName, nGroups, nResponsesPerGroup,
                    pooling, objectiveFunction,
                    outputDirectory, priorDistribution=prior, nProcesses=1)

    print(trueValueString)
    diagnoseSamples(outputDirectory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example MCMC to sample from Gaussian distribution.")

    parser.add_argument(
        "pooling", nargs="?", default="partial",
        help="Pooling method (optional) : partial, complete or none. " +
        "Default is partial."
    )

    args = parser.parse_args()

    main(args.pooling)
