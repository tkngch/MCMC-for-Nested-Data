#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

    print("True value:")

    distributions = []
    for i, name in enumerate(parameterName):
        mu = numpy.random.normal(loc=0, scale=1, size=nGroups)
        sd = numpy.random.gamma(1)

        tmp = [scipy.stats.norm(loc=loc, scale=sd) for loc in mu]
        distributions.append(tmp)

        print("\t%s: {mean: %.2f, var: %.2f}"
              % (name, numpy.mean(mu), numpy.var(mu)))
    print("")

    groupIndex = [i for i in range(nGroups) for j in range(nResponsesPerGroup)]

    func = functools.partial(_computeLogLikelihood,
                             groupIndex=groupIndex,
                             distributions=distributions)

    prior = [scipy.stats.norm(loc=0, scale=1) for name in parameterName]

    return func, prior


def main():
    nChains = 2
    nIter = 1000
    nSamples = 100
    outputDirectory = "./example/sample/distribution/"

    parameterName = ("a", "b", "c")

    nGroups = 10
    nResponsesPerGroup = 10
    # pooling = "partial"
    pooling = "none"

    objectiveFunction, prior =\
        getFunction(parameterName, nGroups, nResponsesPerGroup)

    samplePosterior(nChains, nIter, nSamples,
                    parameterName, nGroups, nResponsesPerGroup,
                    pooling, objectiveFunction,
                    outputDirectory, priorDistribution=prior, nProcesses=2)

    diagnoseSamples(outputDirectory)


if __name__ == "__main__":
    main()
