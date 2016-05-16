#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import numpy
import scipy.stats
from posteriorSampling import samplePosterior
# from diagnosis import diagnose
numpy.random.seed(12345)


def getFunction(parameterName, nGroups, nResponsesPerGroup):

    def _computeLogLikelihood(
        parameter, nGroups, nResponsesPerGroup, distributions):

        ll = numpy.zeros(nGroups * nResponsesPerGroup)
        i = 0
        for j in range(nGroups):
            for k in range(nResponsesPerGroup):
                ll[i] = sum([dist[j].logpdf(parameter[n][j])
                             for n, dist in enumerate(distributions)])
                i += 1

        return ll

    print("True value:")

    distributions = []
    for i, name in enumerate(parameterName):
        scale = numpy.random.random() * 1

        mu = [(i + 1) * scale + j / float(nGroups) for j in range(nGroups)]
        sd = numpy.sqrt(abs(scale))

        tmp = [scipy.stats.norm(loc=loc, scale=sd) for loc in mu]
        distributions.append(tmp)

        print("\t%s: {mean: %.2f, scale: %.2f}"
              % (name, numpy.mean(mu), scale))
    print("")

    func = functools.partial(_computeLogLikelihood,
                             nGroups=nGroups,
                             nResponsesPerGroup=nResponsesPerGroup,
                             distributions=distributions)

    return func


def main():
    nChains = 2
    nIter = 100
    nSamples = 100
    outputDirectory = "./example/sample/distribution/"

    parameterName = ("a", "b", "c",)

    nGroups = 10
    nResponsesPerGroup = 10
    pooling = "partial"

    objectiveFunction = getFunction(parameterName, nGroups, nResponsesPerGroup)

    samplePosterior(nChains, nIter, nSamples,
                    parameterName, nGroups, nResponsesPerGroup,
                    pooling, objectiveFunction,
                    outputDirectory)

    # diagnose(outputdir)


if __name__ == "__main__":
    main()
