#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016 Takao Noguchi (tkngch@runbox.com)

"""

MCMC for nested data

This file declares and defines the classes and the functions to sample a
posterior distribution with partial-, complete- or no-pooling method.

"""

import abc
import datetime
import logging
import multiprocessing
from multiprocessing import Process
import os.path
import shutil
import sys

import numpy
import scipy.stats
import scipy.optimize


def samplePosterior(nChains, nIter, nSamples,
                    parameterName, nGroups, nResponsesPerGroup,
                    pooling, logLikelihoodFunction,
                    outputDirectory,
                    saveLogLikelihood=True,
                    priorDistribution=None,
                    startWithMLE=False, startingPointValueRange=None,
                    nProcesses=1, displayProgress=True, loggingLevel="info"):
    """

    :Arguments:

        - nChains : int

        - nIter : int
            Number of iterations per chain

        - nSamples : int
            Number of retained samples per chain

        - parameterName : tuple of str
            e.g., ("alpha", "beta")

        - nGroups : int

        - nResponsesPerGroup : int or list of int

        - pooling : str
            "partial" for hierarchical Bayes. Alternatively, "complete", or
            "none".

        - logLikelihoodFunction : def
            see doc string for HierarchicalBayes

        - outputDirectory : str
            Full path specifying where to save MCMC samples.

        - startingPointValueRange : dict

            The parameter range to initialize a starting point. This is
            ignored during MCMC sampling.

            e.g., {"alpha": [0, 1], "beta": [2, 5]}

        - saveLogLikelihood (optional) : bool, defalt = True
            Whether to save loglikelihood when saving samples. Saved
            loglikelihood can be used to compute waic or loo with loo package
            in R.

        - priorDistribution (optional) : tuple of scipy.stats distributions
            Required for complete and no pooling. Ignored for partial pooling.

        - startWithMLE (optional) : bool, default = True
            Whether to start a chain with maximum likelihood estimation.
            This estimation pools groups and uses Nelder-Mead method.

        - startingPointValueRange (optional) : dict of list, default = None
            The parameter value range to initialize a starting point. This is
            ignored during posterior sampling. If None, the range is set at
            [-100, 100] for all the parameters.  e.g., {"alpha": [0, 1],
            "beta": [2, 5]}

        - nProcesses (optional) : int, default = 1
            How many processes to launch. If 1 (default), multiprocessing is
            not triggered. If zero, all the available cores are used.

        - displayProgress (optional) : bool, default = True
            Whether to display progress.

        - loggingLevel (optional) : str, default = "info"
            "error", "warning", "info", or "debug"

    """

    startTime = datetime.datetime.now()

    if os.path.exists(outputDirectory):
        shutil.rmtree(outputDirectory)

    sampleDirectory = outputDirectory + "/sample/"
    os.makedirs(sampleDirectory, exist_ok=True)

    logDirectory = outputDirectory + "/log/"
    os.makedirs(logDirectory, exist_ok=True)

    logFile = logDirectory + "samplePosterior.log"
    logName = "samplePosterior"
    logger = _getLogger(logFile, logName, loggingLevel)

    msg = "MCMC sampling.\n"
    msg += "\tpooling: %s.\n" % pooling
    msg += "\tnChains: %i, nIterPerChain: %i, nSamplesPerChain: %i."\
        % (nChains, nIter, nSamples)
    logger.info(msg)
    if displayProgress:
        print(msg)

    if nProcesses <= 0:
        nProcesses = min(nChains, multiprocessing.cpu_count())

    if nProcesses == 1:
        for chain in range(nChains):
            _sample(
                chain, nIter, nSamples,
                parameterName, nGroups, nResponsesPerGroup, pooling,
                logLikelihoodFunction, sampleDirectory, saveLogLikelihood,
                priorDistribution, startWithMLE, startingPointValueRange,
                displayProgress, loggingLevel, logDirectory)

    elif nProcesses > 1:
        activeProcesses = []
        for chain in range(nChains):
            _displayProgress = displayProgress and (len(activeProcesses) == 0)
            process = Process(target=_sample,
                              args=(chain, nIter, nSamples,
                                    parameterName, nGroups,
                                    nResponsesPerGroup, pooling,
                                    logLikelihoodFunction, sampleDirectory,
                                    saveLogLikelihood, priorDistribution,
                                    startWithMLE, startingPointValueRange,
                                    _displayProgress, loggingLevel,
                                    logDirectory))
            process.start()
            activeProcesses.append(process)

            if len(activeProcesses) % nProcesses == 0:
                for activeProcess in activeProcesses:
                    activeProcess.join()
                activeProcesses.clear()

    else:
        print("Invalid number of processes: ", nProcesses)
        print("Exiting.")
        sys.exit()

    endTime = datetime.datetime.now()
    elapsed = endTime - startTime
    msg = "Finished. The elapsed time in total is %s."\
        % datetime.timedelta(seconds=int(elapsed.total_seconds()))

    logger.info(msg)
    if displayProgress:
        print("")
        _printProgress(msg)


def _sample(chain, nIter, nSamples,
            parameterName, nGroups, nResponsesPerGroup, pooling,
            logLikelihoodFunction, sampleDirectory, saveLogLikelihood,
            priorDistribution, startWithMLE, startingPointValueRange,
            displayProgress, loggingLevel, logDirectory):

    seed = chain
    mcmc = MCMC(chain, seed, nIter, nSamples,
                parameterName, nGroups, nResponsesPerGroup, pooling,
                logLikelihoodFunction, sampleDirectory, saveLogLikelihood,
                priorDistribution, startWithMLE, startingPointValueRange,
                displayProgress, loggingLevel, logDirectory)
    mcmc.run()


class Parameter(object):
    def __init__(self, parameterName, index, value, prior, proposalSd, logger):
        """

        Parameter class defines, well, parameter. Users are not expected to
        directly interact with this class.

        :Arguments:

            - parameterName : string

            - index : int
                This is used as a group index, and the unique name of the
                parameter is set to "%s[%.3i]" % (parameterName, index).

            - value : double
                Initial value of parameter. If None, a random sample from prior
                is taken.

            - prior : scipy.stats distribution

            - proposalSd : double

            - logger

        """

        self._logger = logger
        self._parameterName = parameterName
        self._uniqueName = "%s[%.3i]" % (parameterName, index)
        self._value = value
        self._logLikelihood = numpy.nan
        self.setPrior(prior)

        self._proposalSd = proposalSd
        self._adaptiveScaleFactor = 1.
        self._nAccepted = 0.
        self._nRejected = 0.

    def setPrior(self, prior):
        self._logger.debug("The new prior has mean of %.3f and var of %.3f."
                           % (prior.mean(), prior.var()))
        self._prior = prior

        if self._value is None:
            self.samplePriorAndSetValue()

        self._updateLogPrior()
        self._updateLogPosterior()

    def samplePriorAndSetValue(self):
        self._value = self._samplePrior()

    def _samplePrior(self):
        return self._prior.rvs()

    def _updateLogPrior(self):
        self._logPrior = self.getLogPrior(self._value)

    def getLogPrior(self, value):
        return self._prior.logpdf(value)

    @property
    def header(self):
        return self._uniqueName

    @property
    def value(self):
        return self._value

    def propose(self):
        sd = self._proposalSd * self._adaptiveScaleFactor
        self._proposal = numpy.random.normal(self._value, sd)

    @property
    def proposal(self):
        return self._proposal

    @property
    def logPrior(self):
        return self._logPrior

    @property
    def logPosterior(self):
        return self._logPosterior

    @property
    def logLikelihood(self):
        return self._logLikelihood

    @logLikelihood.setter
    def logLikelihood(self, val):
        self._logger.debug("%s ll updated from %f to %f"
                           % (self._uniqueName, self._logLikelihood, val))
        self._logLikelihood = val
        self._updateLogPosterior()

    def _updateLogPosterior(self):
        self._logPosterior = self._logPrior + self._logLikelihood

    def step(self, proposalLogLikelihood):
        proposalLogPrior = self.getLogPrior(self._proposal)
        proposalLogPosterior = proposalLogPrior + proposalLogLikelihood

        msg = "%s\n" % self._uniqueName
        msg += "\tvalue: %.3f, logLikelihood %.3f, logPosterior: %.3f\n"\
            % (self._value, self._logLikelihood, self._logPosterior)
        msg += "\tproposal: %.3f, logll: %.3f, logPosterior: %.3f\n"\
            % (self._proposal, proposalLogLikelihood, proposalLogPosterior)
        self._logger.debug(msg)

        diff = proposalLogPosterior - self._logPosterior

        if (
            (not numpy.isfinite(self._logPosterior)) and
            numpy.isfinite(proposalLogPosterior)
        ):
            self._accept(proposalLogLikelihood)
            return True

        elif not numpy.isfinite(proposalLogLikelihood):
            self._reject()
            return False

        elif not numpy.isfinite(diff):
            self._reject()
            return False

        elif numpy.log(numpy.random.random()) < diff:
            self._accept(proposalLogLikelihood)
            return True

        self._reject()
        return False

    def _accept(self, proposalLogLikelihood):
        self._value = self._proposal
        self._logLikelihood = proposalLogLikelihood

        self._updateLogPrior()
        self._updateLogPosterior()

        self._nAccepted += 1.
        self._logger.debug("%s\taccepted" % self._uniqueName)
        self._proposal = None

    def _reject(self):
        self._nRejected += 1.
        self._logger.debug("%s\trejected" % self._uniqueName)
        self._proposal = None

    def tune(self):
        """
        Tunes the scaling parameter for the proposal distribution
        according to the acceptance rate
        """

        # Flag for tuning state
        tuning = True

        # Calculate recent acceptance rate
        if not (self._nAccepted + self._nRejected):
            return tuning
        acceptanceRate = self._nAccepted / (self._nAccepted + self._nRejected)

        currentFactor = self._adaptiveScaleFactor

        # Switch statement
        if acceptanceRate < 0.001:
            # reduce by 90 percent
            self._adaptiveScaleFactor *= 0.1
        elif acceptanceRate < 0.05:
            # reduce by 50 percent
            self._adaptiveScaleFactor *= 0.5
        elif acceptanceRate < 0.2:
            # reduce by ten percent
            self._adaptiveScaleFactor *= 0.9
        elif acceptanceRate > 0.95:
            # increase by factor of ten
            self._adaptiveScaleFactor *= 10.0
        elif acceptanceRate > 0.75:
            # increase by double
            self._adaptiveScaleFactor *= 2.0
        elif acceptanceRate > 0.5:
            # increase by ten percent
            self._adaptiveScaleFactor *= 1.1
        else:
            tuning = False

        # reset rejection count
        self._nAccepted = 0.
        self._nRejected = 0.

        # Prevent from tuning to zero
        if not self._adaptiveScaleFactor:
            self._adaptiveScaleFactor = currentFactor
            return False

        msg = "%s\t" % self._uniqueName
        msg += "acceptance rate: %.2f\t" % acceptanceRate
        msg += "adaptive scale factor: %.2f" % self._adaptiveScaleFactor
        self._logger.debug(msg)

        return tuning


class HyperParameter(object):
    def __init__(self, parameterName, start, logger):
        """

        This class defines hyper-parameter. Prior for hyper-parameter is a
        uniform for mean and sd. See page 288 on BDA3.

        As with Parameter class, users are not expected to directly interact
        with this class.

        :Arguments:
            - parameterName : str

            - start : dict

            - verbose : bool
        """

        self._parameterName = parameterName
        self._value = start
        self._hyperParameterName = [name for name in self._value]
        self._logger = logger

    def update(self, x):
        """
        Gibbs sampling.
        """
        if type(x) == list:
            x = numpy.array(x)
        if type(x) != numpy.ndarray:
            raise Exception("Wrong type is used for hyperparameter update: %s"
                            % type(x))

        self._updateMean(x)
        self._updateVar(x)

        msg = "%s\n" % self._parameterName
        msg += "\tcurrent: %s\n" % self.value.__str__()
        msg += "\tupdated: %s\n" % self.value.__str__()
        self._logger.debug(msg)

    def _updateMean(self, x):
        """
        See page 289 on BDA3.
        """
        muHat = numpy.mean(x)  # eq. 11.13
        sd = numpy.sqrt(self._value["sigma2"] / len(x))  # eq.11.12
        self._value["mu"] = numpy.random.normal(muHat, sd)  # eq. 11.12

    def _updateVar(self, x):
        """
        See pages 289-290 on BDA3.
        """
        n = len(x)
        hat = numpy.sum((x - self._value["mu"]) ** 2) / (n - 1)  # eq.11.17
        self._value["sigma2"] = self._sampleInvChisq(n - 1, hat)  # eq. 11.16

    def _sampleInvChisq(self, v, s2):
        return scipy.stats.invgamma(v / 2., scale=(v / 2.) * s2).rvs()

    def get_distribution(self):
        return scipy.stats.norm(loc=self._value["mu"],
                                scale=numpy.sqrt(self._value["sigma2"]))

    @property
    def header(self):
        return ["%s_%s" % (self._parameterName, x)
                for x in self._hyperParameterName]

    @property
    def value(self):
        return [self._value[name] for name in self._hyperParameterName]


class StepMethod(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, parameterName, startingPoint,
                 nGroups, nResponsesPerGroup, logLikelihoodFunction,
                 priorDistribution, logger):
        """
        This class defines the step method for hierarchical Bayes.

        :Arguments:

            - parameterName : tuple of str
                e.g., ("alpha", "beta")

            - startingPoint : list
                List of parameter values, one for each parameter. All groups
                are assigned with the same parameter values. Its first value is
                the value for the first parameter in parameterName above, so
                the length has to be the same as the length of parameterName
                above.
                e.g., [1, 2]

            - nGroups : int

            - nResponsesPerGroup : list of int

            - logLikelihoodFunction : def

                A function that accepts a nested list of parameter values and
                returns a list of log likelihood.  No other arguments are
                passed to this function. If your log likelihood function
                requires any other argument, consider partial application
                (functools.partial).

                The list of parameter values is organized as:
                    [[the first parameter for the first group,
                      the first parameter for the first group,
                      ...
                      the first parameter for the first group,
                      the first parameter for the second group,
                      ...
                      the first parameter for the last group],
                     [the second parameter for the first group,
                      the second parameter for the first group,
                      ...
                      the second parameter for the first group,
                      the second parameter for the second group,
                      ...
                      the second parameter for the last group],
                     ...
                    ]

                The parameter values for each group are repeated according to
                nResponsesPerGroup.

                Also, the order of parameters is as in the parameterName
                argument above (e.g., if parameterName = ("alpha", "beta"), the
                first parameter is alpha, the second is beta).

                The output value should be:
                    [ll for the first group's first response,
                     ll for the first group's second response,
                     ...,
                     ll for the first group's last response,
                     ll for the second group's first response,
                     ll for the second group's second response,
                     ...,
                     ll for the second group's last response,
                     ll for the third group's first response,
                     ...]

            - priorDistribution (optional) : tuple of scipy.stats distributions

            - logger
        """
        self._parameterName = parameterName
        self._nGroups = nGroups
        self._priorDistribution = priorDistribution
        self._logger = logger

        self._nResponses = sum(nResponsesPerGroup)

        self._groupSwitchPoint =\
            numpy.hstack([0, numpy.cumsum(nResponsesPerGroup)])
        self._groupIndex = [0] * self._nResponses
        group = 0
        for i in range(self._nResponses):
            while True:
                if (
                        (self._groupSwitchPoint[group] <= i) and
                        (i < self._groupSwitchPoint[group + 1])
                ):
                    break
                else:
                    group += 1

                if self._nGroups <= group:
                    raise Exception()

            self._groupIndex[i] = group
        self._logger.debug("groupSwitchPoint: %s" %
                           self._groupSwitchPoint.__str__())
        self._logger.debug("groupIndex: %s" % self._groupIndex.__str__())

        nParameters = len(self._parameterName)
        self._parameterForLlFunction = [[0] * self._nResponses] * nParameters
        self._logLikelihoodFunction = logLikelihoodFunction

        assert(len(parameterName) == len(startingPoint))
        self._parameter = {}
        self._setStartingPoint(startingPoint)

    def _setStartingPoint(self, startingPoint):
        for i, name in enumerate(self._parameterName):
            proposalSd = 1.
            prior = self._priorDistribution[name]
            value = startingPoint[i]

            self._parameter[name] =\
                [Parameter(name, j, value, prior, proposalSd, self._logger)
                 for j in range(self._nGroups)]

    def step(self, tune):
        for name in self._parameterName:
            self._stepOneParameter(name, tune)
            self._stepHyperParameter(name)

    def _stepOneParameter(self, name, tune):
        for i in range(self._nGroups):
            self._parameter[name][i].propose()

        llarray = self._computeParameterLogLikelihood(name)

        for i, ll in enumerate(llarray):
            accepted = self._parameter[name][i].step(ll)

            if accepted:
                for name_ in self._parameterName:
                    self._parameter[name_][i].logLikelihood = ll

            if tune:
                self._parameter[name][i].tune()

    def _computeLogLikelihood(self, proposedParameter):
        for i, name in enumerate(self._parameterName):
            if name == proposedParameter:
                self._parameterForLlFunction[i] =\
                    [self._parameter[name][j].proposal
                     for j in self._groupIndex]
            else:
                self._parameterForLlFunction[i] =\
                    [self._parameter[name][j].value for j in self._groupIndex]

        ll = self._logLikelihoodFunction(self._parameterForLlFunction)
        assert(len(ll) == self._nResponses)
        return ll

    def _computeParameterLogLikelihood(self, proposedParameter):
        ll = self._computeLogLikelihood(proposedParameter)
        groupLL = [
            sum(ll[self._groupSwitchPoint[i]:self._groupSwitchPoint[i + 1]])
            for i in range(self._nGroups)]

        return groupLL

    def _stepHyperParameter(self, name):
        pass

    @property
    def header(self):
        header = []
        for name in self._parameterName:
            for parameter in self._parameter[name]:
                header.append(parameter.header)
        return header

    @property
    def values(self):
        values = []
        for name in self._parameterName:
            for parameter in self._parameter[name]:
                values.append(parameter.value)
        return values

    @property
    def logLikelihood(self):
        proposedParameter = None
        return self._computeLogLikelihood(proposedParameter)


class CompletePooling(StepMethod):
    def __init__(self, parameterName, startingPoint,
                 nGroups, nResponsesPerGroup, logLikelihoodFunction,
                 priorDistribution, logger):

        _nGroups = 1
        if type(nResponsesPerGroup) == int:
            _nResponsesPerGroup = [nResponsesPerGroup * nGroups]
        elif type(nResponsesPerGroup) == list:
            _nResponsesPerGroup = [sum(nResponsesPerGroup)]

        if (
                (priorDistribution is None) or
                (len(parameterName) != len(priorDistribution))
        ):
            raise ValueError("Invalid prior")

        _priorDistribution = \
            dict((key, dist)
                 for key, dist in zip(parameterName, priorDistribution))

        super().__init__(parameterName, startingPoint,
                         _nGroups, _nResponsesPerGroup, logLikelihoodFunction,
                         _priorDistribution, logger)


class NoPooling(StepMethod):
    def __init__(self, parameterName, startingPoint,
                 nGroups, nResponsesPerGroup, logLikelihoodFunction,
                 priorDistribution, logger):

        if (
                (priorDistribution is None) or
                (len(parameterName) != len(priorDistribution))
        ):
            raise ValueError("Invalid prior")

        _priorDistribution = \
            dict((key, dist)
                 for key, dist in zip(parameterName, priorDistribution))

        super().__init__(parameterName, startingPoint,
                         nGroups, nResponsesPerGroup, logLikelihoodFunction,
                         _priorDistribution, logger)


class PartialPooling(StepMethod):
    def __init__(self, parameterName, startingPoint,
                 nGroups, nResponsesPerGroup, logLikelihoodFunction,
                 priorDistribution, logger):

        if priorDistribution is not None:
            logger.info("Partial pooling ignores prior distribution.")
        _priorDistribution = None

        super().__init__(parameterName, startingPoint,
                         nGroups, nResponsesPerGroup, logLikelihoodFunction,
                         _priorDistribution, logger)

    def _setStartingPoint(self, startingPoint):
        self._initialiseParameters(startingPoint)
        self._determineIndividualStartingPoint()

    def _initialiseParameters(self, startingPoint):
        for i, name in enumerate(self._parameterName):
            val = startingPoint[i]

            start = {"mu": val,
                     "sigma2": numpy.sqrt(numpy.abs(val) / 10.)}

            startStr = ", ".join(["%s: %.8f" % (key, start[key])
                                  for key in start])
            self._logger.info("Staring state:\t %s {%s}" % (name, startStr))

            self._parameter[name + "_hyper"] = \
                HyperParameter(name, start, self._logger)

            prior = self._getParameterPrior(name)
            proposalSd = 1.
            value = None
            self._parameter[name] =\
                [Parameter(name, j, value, prior, proposalSd, self._logger)
                 for j in range(self._nGroups)]

    def _determineIndividualStartingPoint(self):
        ll = [-1 * float("inf")] * self._nGroups

        proposedParameter = None
        while not all(numpy.isfinite(ll)):
            ll = self._computeParameterLogLikelihood(proposedParameter)

            for name in self._parameterName:
                for i in range(self._nGroups):
                    if numpy.isfinite(ll[i]):
                        self._parameter[name][i].logLikelihood = ll[i]
                    else:
                        self._parameter[name][i].samplePriorAndSetValue()

    def _getParameterPrior(self, name):
        return self._parameter[name + "_hyper"].get_distribution()

    def _stepHyperParameter(self, name):
        values = [self._parameter[name][i].value for i in range(self._nGroups)]
        self._parameter[name + "_hyper"].update(values)

        prior = self._getParameterPrior(name)
        for i in range(self._nGroups):
            self._parameter[name][i].setPrior(prior)

    @property
    def header(self):
        header = []
        for name in self._parameterName:
            header.extend(self._parameter[name + "_hyper"].header)
            for parameter in self._parameter[name]:
                header.append(parameter.header)
        return header

    @property
    def values(self):
        values = []
        for name in self._parameterName:
            values.extend(self._parameter[name + "_hyper"].value)
            for parameter in self._parameter[name]:
                values.append(parameter.value)
        return values


class Sampler(object):
    def __init__(self, stepMethod, chain, sampleFile, llFile,
                 saveLogLikelihood, displayProgress, logger):
        """
        Sampler class to sample from posterior.

        When sampleDirectory is supplied, posterior samples and their log
        likelihoods are saved onto files. The log likelihood files are for
        computing the widely applicable information criteria and the
        leave-one-out cross-validation estimate, using loo package in R.

        :Arguments:

            - stepMethod : StepMethod class object

            - sampleDirectory : str
                Where to save posterior samples. When it is None, samples are
                not saved into a file.

            - saveLogLikelihood : bool
                Whether to save loglikelihood when saving samples. Saved
                loglikelihood can be used to compute waic or loo with loo
                package in R.

            - displayProgress : bool
                Whether to display progress and estimated time of termination.

            - logger
        """

        self._stepMethod = stepMethod
        self._chain = chain
        self._outputFile = {"sample": sampleFile, "loglikelihood": llFile}
        self._saveLogLikelihood = saveLogLikelihood
        self._displayProgress = displayProgress
        self._logger = logger

    def sample(self, iter, burn, thin, tuneInterval):
        """
        sample(iter, burn, thin, tuneInterval, verbose)

        :Arguments:

          - iter : int
                Total number of iterations to do

          - burn : int
                Variables will not be tallied until this many iterations are
                complete.

          - thin : int
                Variables will be tallied at intervals of this many iterations.

          - tuneInterval (optional) : int, default = 100
                Step methods will be tuned at intervals of this many
                iterations.

        """

        iter, burn, thin = numpy.floor([iter, burn, thin]).astype(int)

        if burn > iter:
            raise ValueError("Burn interval cannot be larger than iterations.")

        self._iter = int(iter)
        self._burn = int(burn)
        self._thin = int(thin)
        self._tuneInterval = int(tuneInterval)

        # Run sampler
        self._loop()

    def _loop(self):
        """
        Draws samples from the posterior.
        """

        self._startTime = datetime.datetime.now()
        loggingInterval = int(numpy.round(self._iter / 10.))

        tune = False
        # Loop
        for i in range(self._iter):

            # Tune at interval
            if i and (i < self._burn) and (i % self._tuneInterval == 0):
                tune = True
            else:
                tune = False

            # take a step
            self._stepMethod.step(tune)

            if i == self._burn:
                self._printHeader()

            # Record sample to trace, if appropriate
            if i % self._thin == 0 and i >= self._burn:
                self._printSample(i)

                if self._saveLogLikelihood:
                    self._printLogLikelihood()

            if i % loggingInterval == 0:
                self._logProgress(i)

        self._logProgress(self._iter)

    def _printHeader(self):
        header = "index,chain," + ",".join(self._stepMethod.header)
        self._print("sample", header)

    def _printSample(self, i):
        out = "%i,%i," % (i, self._chain)
        out += ",".join(["%f" % v for v in self._stepMethod.values])
        self._print("sample", out)

    def _printLogLikelihood(self):
        out = ",".join(["%f" % v for v in self._stepMethod.logLikelihood])
        self._print("loglikelihood", out)

    def _logProgress(self, i):
        now = datetime.datetime.now()

        if i == 0:
            msg = r"Sampling started. 0% complete."
        elif i >= self._iter:
            elapsed = now - self._startTime
            msg = "100%% complete. Elapsed Time: %s."\
                % datetime.timedelta(seconds=int(elapsed.total_seconds()))
        else:
            percentage = float(i) / self._iter
            remain = (1 - percentage) * (now - self._startTime) / percentage
            msg = "%i%% complete. ETA: %s."\
                % (percentage * 100, str(_getStrfTime(now + remain)))

        self._printProgress(msg)

    def _printProgress(self, msg):
        self._logger.info("chain %i. %s" % (self._chain, msg))
        if self._displayProgress:
            _printProgress(msg)

    def _print(self, destination, output):
        with open(self._outputFile[destination], "a") as h:
            h.write(output)
            h.write("\n")


class MCMC(object):
    """
    This class defines StepMethod and runs Sampler.
    """

    def __init__(self,
                 chain, seed, nIter, nSamples,
                 parameterName, nGroups, nResponsesPerGroup, pooling,
                 logLikelihoodFunction,
                 sampleDirectory, saveLogLikelihood,
                 priorDistribution, startWithMLE, startingPointValueRange,
                 displayProgress, loggingLevel, logDirectory):
        """

        :Arguments:

            - chain : int
                ID number for MCMC chain

            - seed : int
                seed for random number generator

            - nIter : int
                Number of iteration per chain

            - nSamples: int
                Number of retained samples per chain

            - parameterName : tuple of str
                e.g., ("alpha", "beta")

            - nGroups : int

            - nResponsesPerGroup : int or list of int

            - pooling : str
                "partial" for hierarchical Bayes. Alternatively, "complete", or
                "none".

            - logLikelihoodFunction : def
                see doc string for HierarchicalBayes

            - sampleDirectory : str
                Full path specifying where to save MCMC samples.

            - saveLogLikelihood : bool, defalt = True
                Whether to save loglikelihood when saving samples. Saved
                loglikelihood can be used to compute waic or loo with loo
                package in R.

            - priorDistribution (optional) : tuple of scipy.stats distributions
                Required for complete and no pooling. Ignored for partial
                pooling.

            - startWithMLE : bool, default = True
                Whether to start a chain with maximum likelihood estimation.
                This estimation pools groups and uses Nelder-Mead method.

            - startingPointValueRange : dict, default = None
                The parameter value range to initialize a starting point. This
                is ignored during posterior sampling. If None, the range is set
                at [-100, 100] for all the parameters.
                e.g., {"alpha": [0, 1], "beta": [2, 5]}

            - displayProgress : bool, default = True
                Whether to display progress.

            - loggingLevel : str

            - logDirectory : str

        """

        if displayProgress:
            print("\n- Chain %i -" % chain)

        numpy.random.seed(seed)

        self._chain = chain
        if nIter < nSamples:
            print("nIter (%i) cannot be less than nSamples (%i)." %
                  (nIter, nSamples))
            raise Exception()
        elif nIter // 2 > nSamples:
            self._burn = nIter // 2
        else:
            self._burn = nIter - nSamples

        self._thin = int(numpy.ceil((nIter - self._burn) / nSamples))
        self._nIter = nIter
        self._nSamples = nSamples

        self._parameterName = parameterName
        self._nGroups = nGroups

        if type(nResponsesPerGroup) == int:
            self._nResponsesPerGroup = [nResponsesPerGroup] * nGroups
        elif type(nResponsesPerGroup) == list:
            self._nResponsesPerGroup = nResponsesPerGroup
        self._nResponses = sum(self._nResponsesPerGroup)

        if pooling in ("partial", "none", "complete"):
            self._pooling = pooling
        else:
            raise Exception("Invalid pooling: ", pooling)

        self._logLikelihoodFunction = logLikelihoodFunction
        self._priorDistribution = priorDistribution

        self._sampleFile = sampleDirectory + "/sample.%i.csv" % self._chain
        self._llFile = sampleDirectory + "/logLikelihood.%i.csv" % self._chain

        self._saveLogLikelihood = saveLogLikelihood
        self._displayProgress = displayProgress or False

        logFile = logDirectory + "/mcmc.chain%.2i.log" % self._chain
        logName = "mcmc.chain%.2i" % self._chain
        self._logger = _getLogger(logFile, logName, loggingLevel)

        self._findStartingPoint(startWithMLE, startingPointValueRange)

    def _findStartingPoint(self, startWithMLE, startingPointValueRange):
        self._printProgress("Started looking for a reasonable starting state.")

        if startingPointValueRange is None:
            startingPointValueRange = {}

        ll = numpy.inf
        x = [0] * len(self._parameterName)
        counter = 0

        while not numpy.isfinite(ll):
            for i, name in enumerate(self._parameterName):
                if name in startingPointValueRange:
                    x[i] = numpy.random.uniform(
                        low=startingPointValueRange[name][0],
                        high=startingPointValueRange[name][1])
                elif self._priorDistribution is not None:
                    x[i] = self._priorDistribution[i].rvs()
                else:
                    x[i] = numpy.random.norm()

            ll = self._mleObjectiveFunction(x)

            counter += 1
            if (counter > 10):
                raise RuntimeError("Failed to find a valid starting state.")

        self._startingPoint = x

        self._printProgress("Found a reasonable starting state.")

        if startWithMLE:
            self._optimizeStartingPoint()

    def _printProgress(self, msg):
        self._logger.info("chain %i. %s" % (self._chain, msg))
        if self._displayProgress:
            _printProgress(msg)

    def _mleObjectiveFunction(self, x):
        param = [[p for i in range(self._nResponses)] for p in x]
        ll = self._logLikelihoodFunction(param)
        return -1 * numpy.sum(ll)

    def _optimizeStartingPoint(self):
        self._printProgress("Started optimising a starting state.")

        optimised = False
        n = 0
        while not optimised:
            n += 1
            res = scipy.optimize.minimize(self._mleObjectiveFunction,
                                          self._startingPoint,
                                          method="Nelder-Mead",
                                          options={"maxiter": None,
                                                   "maxfev": None,
                                                   "xtol": 0.0001,
                                                   "ftol": 0.0001})

            self._logger.info("\t%s %.2f." % (res.message, res.fun))

            if numpy.isfinite(res.fun):
                self._startingPoint = res.x

                if res.success:
                    optimised = True

            else:
                self._findStartingPoint()

            if n > 10:
                msg = "Could not find a local maxima. "
                msg += "A random point is taken as a starting state."
                self._logger.warn(msg)

                self._startingPoint = res.x
                optimised = True

        self._printProgress("Optimised a starting state.")

    def run(self):
        self._printProgress("Determining individual starting points.")

        stepMethod = {"partial": PartialPooling,
                      "none": NoPooling,
                      "complete": CompletePooling}[self._pooling](
                    self._parameterName, self._startingPoint, self._nGroups,
                    self._nResponsesPerGroup, self._logLikelihoodFunction,
                    self._priorDistribution, self._logger)

        sampler = Sampler(
            stepMethod, self._chain, self._sampleFile, self._llFile,
            self._saveLogLikelihood, self._displayProgress, self._logger)

        tuneInterval = 100
        sampler.sample(self._nIter, self._burn, self._thin, tuneInterval)


def _getLogger(logFile, logName, loggingLevel):
    if loggingLevel == "debug":
        level = logging.DEBUG
    elif loggingLevel == "info":
        level = logging.INFO
    elif loggingLevel == "warning":
        level = logging.WARNING
    elif loggingLevel == "error":
        level = logging.ERROR

    logger = logging.getLogger(logName)
    logger.setLevel(level)

    handler = logging.FileHandler(logFile)
    handler.setLevel(level)

    logFormat = "%(asctime)s - %(name)s - %(levelname)s\n%(message)s\n"
    formatter = logging.Formatter(logFormat)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


def _printProgress(msg):
    now = _getStrfTime(datetime.datetime.now())
    print(now + "\t" + msg)


def _getStrfTime(time):
    datetimefmt = "%Y/%m/%d %H:%M:%S"
    return time.strftime(datetimefmt)
