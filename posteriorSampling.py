#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016 Takao Noguchi (tkngch@runbox.com)

"""

MCMC with Hierarchical Bayes

This file declares and defines the classes and the functions to sample a
posterior distribution using MCMC algorithm.

"""

import shutil
import sys
import os.path
import datetime
import multiprocessing
from multiprocessing import Process
import numpy
import scipy.stats
import scipy.optimize


# ---------------- #
# global variables #
# ---------------- #
DATETIMEFMT = "%H:%M:%S on %d/%m/%Y"


# ---------------------------- #
# class declaration/definition #
# ---------------------------- #

class Parameter(object):
    def __init__(self, parameterName, index, prior,
                 proposalSd=1., verbose=False):
        """

        Parameter class defines, well, parameter. Users are not expected to
        directly interact with this class.

        :Arguments:

            - parameterName : string

            - index : int
                This is used as a group index, and the unique name of the
                parameter is set to "%s[%.3i]" % (parameterName, index).

            - prior : scipy.stats distribution

            - proposalSd (optional) : double, default = 1.

            - verbose (optional) : bool, default = False

        """

        self._verbose = verbose or False
        self._parameterName = parameterName
        self._uniqueName = "%s[%.3i]" % (parameterName, index)
        self._family = family
        self._value = None
        self._logLikelihood = numpy.nan
        self._setPrior(prior)

        self._proposalSd = proposalSd
        self._adaptiveScaleFactor = 1.
        self._nAccepted = 0.
        self._nRejected = 0.

    def _setPrior(self, prior):
        if self._verbose:
            print("The new prior has mean of %.3f and var of %.3f."
                  % (prior.mean(), prior.var()))
        self._prior = prior

        if self._value is None:
            self.samplePriorAndSetValue()

        self._updateLogPrior()
        self._updateLogPosterior()

    def samplePriorAndSetValue(self):
        self._value = self._samplePrior()

    def _samplePrior(self):
        if "log" in self._family:
            value = numpy.exp(self._prior.rvs())
        else:
            value = self._prior.rvs()

        if "negated" in self._family:
            value *= -1

        return value

    def _updateLogPrior(self):
        self._logPrior = self.getLogPrior(self._value)

    def getLogPrior(self, value):
        if "negated" in self._family:
            value = -1 * value
        if "log" in self._family:
            value = numpy.log(value)

        if "poisson" in self._family:
            logPrior = self._prior.logpmf(value)
        else:
            logPrior = self._prior.logpdf(value)

        return logPrior

    @property
    def header(self):
        return self._uniqueName

    @property
    def value(self):
        return self._value

    def propose(self):
        sd = self._proposalSd * self._adaptiveScaleFactor
        self._proposal = numpy.random.normal(self._value, sd)

        if "poisson" in self._family:
            self._proposal = int(round(self._proposal))

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
        if self._verbose:
            print("%s ll updated from %f to %f"
                  % (self._uniqueName, self._logLikelihood, val))
        self._logLikelihood = val
        self._updateLogPosterior()

    def _updateLogPosterior(self):
        self._logPosterior = self._logPrior + self._logLikelihood

    def step(self, proposalLogLikelihood):
        proposalLogPrior = self.getLogPrior(self._proposal)
        proposalLogPosterior = proposalLogPrior + proposalLogLikelihood

        if self._verbose > 0:
            print("%s\t" % self._uniqueName, end="")
            print("value: %.3f, logll: %.3f, logPosterior: %.3f"
                  % (self._value, self._logLikelihood, self._logPosterior),
                  end="\t")
            print("proposal: %.3f, logll: %.3f, logPosterior: %.3f"
                  % (self._proposal, proposalLogLikelihood, proposalLogPosterior),
                  end="\t")

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
        if self._verbose > 0:
            print("accepted")
        self._proposal = None

    def _reject(self):
        self._nRejected += 1.
        if self._verbose > 0:
            print("rejected")
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

        if self._verbose:
            print("%s\t" % self._uniqueName, end="")
            print("acceptance rate: %.2f\t" % acceptanceRate, end="")
            print("adaptive scale factor: %.2f" % self._adaptiveScaleFactor)

        return tuning


class HyperParameter(object):
    def __init__(self, parameterName, start=None, verbose=False):
        """

        This class defines hyper-parameter. Prior for hyper-parameter is a
        uniform for mean and sd.

        As with Parameter class, users are not expected to directly interact
        with this class.

        :Arguments:
            - parameterName : str

            - start (optional) : dict, default = None
                if None, it is set to {"mu": 0, "sigma2": 1}.

            - verbose (optional) : bool, default = False

        """

        self._parameterName = parameterName

        if start is not None:
            self._value = start
        else:
            self._value = {"mu": 0, "sigma2": 1}

        self._hyperParameterName = [name for name in self._value]
        self._verbose = verbose or False

    def update(self, x):
        """
        Gibbs sampling.
        """
        if type(x) == list:
            x = numpy.array(x)
        if type(x) != numpy.ndarray:
            raise Exception("Wrong type is used for hyperparameter update: %s"
                            % type(x))

        if self._verbose:
            print("current: %s" % self.values.__str__(), end="")

        self._updateMean(x)
        self._updateVar(x)

        if self._verbose:
            print("\tupdated: %s" % self.values.__str__())

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


class PartialPooling(object):
    def __init__(self, parameterName, startingPoint,
                 nGroups, nResponsesPerGroup,
                 logLikelihoodFunction,
                 startWithMLE=False, verbose=False):
        """
        This class defines the step method and finds a starting state for MCMC.

        :Arguments:

            - parameterName : tuple of str
                e.g., ("alpha", "beta")

            TODO
            - startingPoint : dict
                e.g., {"alpha": [0, 1], "beta": [2, 5]}

                The parameter range to initialize a starting point. This is
                ignored during MCMC sampling.

            - nGroups : int

            - nResponsesPerGroup : int or list of int

            - logLikelihoodFunction : def

                A function that accepts a nested list of parameter values and
                returns a list of log likelihood.  No other arguments are
                passed to this function. If your log likelihood function
                requires any other argument, consider partial application
                (functools.partial).

                The list of parameter values is organized as:
                    [[0.7, 0.2, ..., 0.3],
                     # nGroups values of the first parameter
                     [4.3, 3.6, ..., 2.9],
                     # nGroups values of the second parameter
                     ...].

                The order of parameters is as in the parameterName argument
                above (e.g., if parameterName = ("alpha", "beta"), the first
                parameter is alpha, the second is beta).

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

            - startWithMLE (optional) : bool, default = False
                Whether to start MCMC with a solution from Nelder-Mead
                optiomization.

            - verbose (optional) : bool, default = False

        """

        self._verbose = verbose or False

        self._parameterName = parameterName
        self._startingPointValueRange = startingPointValueRange

        self._nGroups = nGroups

        if type(nResponsesPerGroup) == int:
            self._nResponsesPerGroup = [nResponsesPerGroup] * nGroups
        elif type(nResponsesPerGroup) == list:
            self._nResponsesPerGroup = nResponsesPerGroup

        self._logLikelihoodFunction = logLikelihoodFunction

        self._llIndex = numpy.hstack(
            [0, numpy.cumsum(self._nResponsesPerGroup)])
        assert(len(self._llIndex) == self._nGroups + 1)

        # TODO
        self._find_starting_point()
        if startWithMLE:
            self._optimise_starting_point()

        if self._verbose:
            print("Starting state:")

        self._initialiseParameters(startingPoint)
        self._determineIndividualStartingPoint()

        if self._verbose:
            print("")

    def _find_starting_point(self):
        if self._verbose:
            print("Started looking for a reasonable starting state at %s." %
                  datetime.datetime.now().strftime(DATETIMEFMT))

        ll = numpy.inf
        x = [0] * len(self._parameterName)

        while not numpy.isfinite(ll):
            for i, name in enumerate(self._parameterName):
                u = numpy.random.uniform(
                    low=self._parameter_value_range[name][0],
                    high=self._parameter_value_range[name][1])

                if self._parameter_family[name] == "poisson":
                    x[i] = int(round(u))
                else:
                    x[i] = u

            ll = self._mle_objective_function(x)

        self._starting_point = x

        if self._verbose:
            print("Found a starting state at %s." %
                  datetime.datetime.now().strftime(DATETIMEFMT),
                  end="\n\n")

    def _mle_objective_function(self, x):
        valid = True
        for i, name in enumerate(self._parameterName):
            if self._parameter_family[name] == "poisson":
                x[i] = int(round(x[i]))

            if (
                (x[i] < self._parameter_value_range[name][0]) or
                (self._parameter_value_range[name][1] < x[i])
            ):
                valid = False

        if not valid:
            return -1 * float("inf")

        param = [[p for i in range(self._nGroups)] for p in x]
        ll = self._logLikelihoodFunction(param)
        return -1 * numpy.sum(ll)

    def _optimise_starting_point(self):
        if self._verbose:
            print("Started optimising a starting state at %s." %
                  datetime.datetime.now().strftime(DATETIMEFMT))

        optimised = False
        n = 0
        while not optimised:
            n += 1
            res = scipy.optimize.minimize(self._mle_objective_function,
                                          self._starting_point,
                                          method="Nelder-Mead",
                                          options={"maxiter": None,
                                                   "maxfev": None,
                                                   "xtol": 0.0001,
                                                   "ftol": 0.0001})

            if self._verbose:
                print("\t%s %.2f." % (res.message, res.fun))

            if numpy.isfinite(res.fun):
                self._starting_point = res.x

                if res.success:
                    optimised = True

            else:
                self._find_starting_point()

            if n > 10:
                if self._verbose:
                    print("Could not find a local maxima. "
                          "A random point is taken as a starting state.")

                self._starting_point = res.x
                optimised = True

        if self._verbose:
            print("Optimised a starting state: %s." %
                  datetime.datetime.now().strftime(DATETIMEFMT))

    def _initialiseParameters(self, startingPoint):
        self._parameter = {}
        for i, name in enumerate(self._parameterName):

            val = startingPoint[i]

            start = {"mu": val,
                     "sigma2": numpy.sqrt(numpy.abs(val) / 10.)}

            if self._verbose:
                startStr = ", ".join(["%s: %.8f" % (key, start[key])
                                      for key in start])
                print("\t %s {%s}" % (name, startStr))

            self._parameter[name + "_hyper"] = \
                HyperParameter(name, family, start=start)

            prior = self._getParameterPrior(name)
            self._parameter[name] = [Parameter(name, j, prior)
                                     for j in range(self._nGroups)]

    def _determineIndividualStartingPoint(self):
        if self._verbose:
            print("Determining individual startin states at %s." %
                  datetime.datetime.now().strftime(DATETIMEFMT))

        ll = [-1 * float("inf")] * self._nGroups

        while not all(numpy.isfinite(ll)):
            ll = self._computeParameterLogLikelihood()

            for name in self._parameterName:
                for i in range(self._nGroups):
                    if numpy.isfinite(ll[i]):
                        self._parameter[name][i].logLikelihood = ll[i]
                    else:
                        self._parameter[name][i].samplePriorAndSetValue()

    def _getParameterPrior(self, name):
        return self._parameter[name + "_hyper"].get_distribution()

    def step(self, tune=False):
        for name in self._parameterName:

            for i in range(self._nGroups):
                self._parameter[name][i].propose()

            llarray = self._computeParameterLogLikelihood(
                proposed_parameter=name)

            for i, ll in enumerate(llarray):
                accepted = self._parameter[name][i].step(ll)

                if accepted:
                    for name_ in self._parameterName:
                        self._parameter[name_][i].logLikelihood = ll

                if tune:
                    self._parameter[name][i].tune()

            values = [self._parameter[name][i].value
                      for i in range(self._nGroups)]
            self._parameter[name + "_hyper"].update(values)

            prior = self._getParameterPrior(name)
            for i in range(self._nGroups):
                self._parameter[name][i].set_prior(prior)

    def _computeParameterLogLikelihood(self, proposed_parameter=None):

        x = [[0] * self._nGroups] * len(self._parameterName)
        for i, name in enumerate(self._parameterName):
            if name == proposed_parameter:
                x[i] = [p.proposal for p in self._parameter[name]]
            else:
                x[i] = [p.value for p in self._parameter[name]]

        ll = self._logLikelihoodFunction(x)

        return [sum(ll[self._llIndex[i]:self._llIndex[i + 1]])
                for i in range(self._nGroups)]

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

    @property
    def logLikelihood(self):
        parameter_values = [[p.value for p in self._parameter[name]]
                            for name in self._parameterName]
        ll = self._logLikelihoodFunction(parameter_values)

        return ll


class Sampler(object):
    def __init__(self, stepMethod, chain, sampleDirectory=None,
                 saveLogLikelihood=True,
                 displayProgress=True, verbose=False):
        """
        Sampler class to sample from posterior.

        When sampleDirectory is supplied, posterior samples and their log
        likelihoods are saved onto files. The log likelihood files are for
        computing the widely applicable information criteria and the
        leave-one-out cross-validation estimate, using loo package in R.

        :Arguments:

            - stepMethod : PartialPooling, NoPooling, or CompletePooling

            - sampleDirectory (optional) : str, default = None
                Where to save posterior samples. When it is None, samples are
                not saved into a file.

            - saveLogLikelihood (optional) : bool, defalt = True
                Whether to save loglikelihood when saving samples. Saved
                loglikelihood can be used to compute waic or loo with loo
                package in R.

            - displayProgress (optional) : bool, default = True
                Whether to display progress and estimated time of termination.

            - verbose (optional) : boolean, default = False

        """

        self._stepMethod = stepMethod
        self._chain = chain
        self._saveLogLikelihood = saveLogLikelihood
        self._displayProgress = displayProgress
        self._verbose = verbose or False

        if sampleDirectory is not None:
            i = 0
            sampleFile = sampleDirectory + "/sample.%i.csv" % i
            llFile = sampleDirectory + "/logLikelihood.%i.csv" % i

            while os.path.isfile(sampleFile):
                i += 1
                sampleFile = sampleDirectory + "/sample.%i.csv" % i
                llFile = sampleDirectory + "/logLikelihood.%i.csv" % i

            # create empty files
            open(sampleFile, "w").close()

            if self._saveLogLikelihood:
                open(llFile, "w").close()
            else:
                llFile = None

        else:
            sampleFile = None
            llFile = None

        self._outputFile = {"sample": sampleFile, "loglikelihood": llFile}

    def sample(self, iter, burn=0, thin=1, tuneInterval=100):
        """
        sample(iter, burn, thin, tuneInterval, verbose)

        :Arguments:

          - iter : int
                Total number of iterations to do

          - burn (optional) : int, default = 0
                Variables will not be tallied until this many iterations are
                complete.

          - thin (optional) : int, default = 1
                Variables will be tallied at intervals of this many iterations.

          - tuneInterval (optional) : int, default = 100
                Step methods will be tuned at intervals of this many
                iterations.

        """

        iter, burn, thin = numpy.floor([iter, burn, thin]).astype(int)

        if burn > iter:
            raise ValueError("Burn interval cannot be larger than iterations.")

        self._n_tally = int(iter) - int(burn)
        self._iter = int(iter)
        self._burn = int(burn)
        self._thin = int(thin)
        self._tuneInterval = int(tuneInterval)

        if self._verbose:
            print("Chain %i. Sampling initialised at %s." %
                  (self._chain,
                   datetime.datetime.now().strftime(DATETIMEFMT)))

        # Run sampler
        self._loop()

    def _loop(self):
        """
        Draws samples from the posterior.
        """

        self._startTime = datetime.datetime.now()
        displayInterval = int(numpy.round(self._iter / 10.))

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
                self._print_header()

            # Record sample to trace, if appropriate
            if i % self._thin == 0 and i >= self._burn:
                self._print_sample(i)

                if self._saveLogLikelihood:
                    self._printLogLikelihood()

            if self._displayProgress and (i % displayInterval == 0):
                self._printProgress(i)

        if self._displayProgress:
            self._printProgress(self._iter)
            print("")

    def _print_header(self):
        header = "index,chain," + ",".join(self._stepMethod.header)
        self._print("sample", header)

    def _print_sample(self, i):
        out = "%i,%i," % (i, self._chain)
        out += ",".join(["%f" % v for v in self._stepMethod.values])
        self._print("sample", out)

    def _printLogLikelihood(self):
        out = ",".join(["%f" % v for v in self._stepMethod.logLikelihood])
        self._print("loglikelihood", out)

    def _printProgress(self, i):
        now = datetime.datetime.now()

        if i == 0:
            self._print(None, "\t  0%% complete at %s." %
                        now.strftime(DATETIMEFMT))
        elif i >= self._iter:
            elapsed = now - self._startTime
            self._print(None, "\t100%% complete at %s. Elapsed Time: %s" %
                        (now.strftime(DATETIMEFMT),
                         datetime.timedelta(
                            seconds=int(elapsed.total_seconds()))))
        else:
            percentage = float(i) / self._iter
            remain = (1 - percentage) * (now - self._startTime) / percentage
            self._print(None,
                        "\t%3i%% complete at %s. ETA: %s." %
                        (percentage * 100, now.strftime(DATETIMEFMT),
                         str((now + remain).strftime(DATETIMEFMT))))

    def _print(self, destination, output):
        if destination is None:
            print(output)
        elif self._outputFile[destination] is None:
            print(output)
        else:
            with open(self._outputFile[destination], "a") as h:
                h.write(output)
                h.write("\n")


class MCMC(object):
    """
    This class defines StepMethod and runs Sampler.
    """

    def __init__(self, parameterName,
                 nGroups, nResponsesPerGroup, logLikelihoodFunction,
                 nChains, nIter, nSamples, chainSeed=None,
                 pooling="partial",
                 startWithMLE=True, startingPointValueRange=None,
                 outputDirectory="./posteriorSample/",
                 clearOutputDirectory=True,
                 # resumePreviousRun=False,
                 nProcesses=None,
                 saveLogLikelihood=True, verbose=True):
        """

        :Arguments:

            - parameterName : tuple of str
                e.g., ("alpha", "beta")

            - nGroups : int

            - nResponsesPerGroup : int or list of int

            - logLikelihoodFunction : def
                see doc string for HierarchicalBayes

            - nIter : int
                Number of iteration per chain

            - nSamples: int
                Number of retained samples per chain

            - nProcesses (optional) : int, default = 1
                How many processes to launch. If 1 (default), multiprocessing
                is not triggered. If zero, all the available cores are used.

            - chainSeed (optional) : iterable, default = None
                Seed for random number generation. The length of the seeds has
                to be the same as the number of chains. If None, range(nChains)
                is used.

            - pooling (optional) : str, default = "partial"
                "partial" for hierarchical Bayes. Alternatively, "complete", or
                "none".

            - startWithMLE (optional) : bool, default = True
                Whether to start a chain with maximum likelihood estimation.
                This estimation pools groups and uses Nelder-Mead method.

            - startingPointValueRange : dict, default = None
                The parameter value range to initialize a starting point. This
                is ignored during posterior sampling. If None, the range is set
                at [-100, 100] for all the parameters.

                e.g., {"alpha": [0, 1], "beta": [2, 5]}

            - outputDirectory : str, default = "./posteriorSample/"
                Full path specifying where to save MCMC samples, their summary,
                and figures (traceplots and bivariate). Before running MCMC,
                everything under this path will be removed.

            - clearOutputDirectory (optional) : bool, default = True
                Whether to delete all the files in the outputdir.

            - saveLogLikelihood (optional) : bool, defalt = True
                Whether to save loglikelihood when saving samples. Saved
                loglikelihood can be used to compute waic or loo with loo
                package in R.

            - verbose (optional) : bool, default = True
                Whether to display progress.

        """

        self._parameterName = parameterName
        self._nGroups = nGroups
        self._nResponsesPerGroup = nResponsesPerGroup
        self._logLikelihoodFunction = logLikelihoodFunction

        if nIter < nSamples:
            print("nIter (%i) cannot be less than nSamples (%i)."
                % (nIter, nSamples))
            raise Exception()
        elif nIter // 2 > nSamples:
            self.burn = nIter // 2
        else:
            self.burn = nIter - nSamples

        self.thin = int(numpy.ceil((nIter - self.burn) / nSamples))
        self._nIter = nIter
        self._nSamples = nSamples

        if nProcesses == 0:
            self.nProcesses = min(nChains, multiprocessing.cpu_count())
        elif nProcesses > 0:
            self._nProcesses = nProcesses
        else:
            print("Invalid nProcesses:", nProcesses)
            raise Exception()

        if chainSeed is None:
            chainSeed = range(nChains)
        elif len(chainSeed) != nChains):
            print("ChainSeed length has to be the same as nChains")
            raise Exception()
        self._chainSeed = chainSeed

        if pooling in ("partial", "none", "complete"):
            self._pooling = pooling
        else:
            print("Invalid pooling: ", pooling)
            raise Exception()

        self._findStartingPoint(startWithMLE, startingPointValueRange)

        if clearOutputDirectory and os.path.exists(outputDirectory):
            shutil.rmtree(outputDirectory)

        self._sampleDirectory = outputDirectory + "/sample/"
        os.makedirs(self._sampleDirectory, exist_ok=True)

        self._saveLogLikelihood = saveLogLikelihood
        self._verbose = verbose or False

    def _findStartingPoint(self, startWithMLE, startingPointValueRange):
        #TODO
        pass

    def sample(self):
        startTime = datetime.datetime.now()

        if self._nProcesses == 1:
            displayProgress = True
            for chain, seed in enumerate(self._chainSeed):
                _sample(seed, self._pooling, self._parameterName,
                        self._startingPoint, self._nGroups,
                        self._nResponsesPerGroup, self._logLikelihoodFunction,
                        chain, self._sampleDirectory, displayProgress,
                        self._verbose)
        else:
            self._parallelSample(stepMethod)

        endTime = datetime.datetime.now()
        elapsed = endTime - startTime
        print("Finished at %s. The elapsed time in total is %s." %
              (endTime.strftime(DATETIMEFMT),
               datetime.timedelta(seconds=int(elapsed.total_seconds()))),
              end="\n\n")


def _sample(self, seed, pooling, parameterName, startingPoint, nGroups,
        nResponsesPerGroup, logLikelihoodFunction, chain, sampleDirectory,
        displayProgress, verbose):

    numpy.random.seed(seed)

    stepMethod = {"partial": PartialPooling,
                    "none": NoPooling,
                    "complete": CompletePooling}[pooling](
                parameterName, startingPoint, nGroups,
                nResponsesPerGroup, logLikelihoodFunction)

    sampler = Sampler(stepMethod, chain,
        sampleDirectory=sampleDirectory,
        saveLogLikelihood=saveLogLikelihood,
        displayProgress=displayProgress,
        verbose=verbose)

    sampler.sample()





def sample_posterior(parameterName, parameter_family, startingPointValueRange,
                     nGroups, nResponsesPerGroup, logLikelihoodFunction,
                     n_chains, n_iter, n_samples, outputdir,
                     prior_for_hyperparameter=None, startWithMLE=True,
                     chain_seed=None, loglikelihood_timeout=60,
                     n_processes=None, clear_outputdir=True,
                     saveLogLikelihood=True, verbose=True):
    """

    :Arguments:

        - parameterName : tuple of str

            e.g., ("alpha", "beta")

        - parameter_family : dict

            e.g., {"alpha": "normal", "beta": "exponential"}

        - startingPointValueRange : dict

            The parameter range to initialize a starting point. This is
            ignored during MCMC sampling.

            e.g., {"alpha": [0, 1], "beta": [2, 5]}

        - nGroups : int

        - nResponsesPerGroup : int or list of int

        - logLikelihoodFunction : def
            see doc string for HierarchicalBayes

        - n_iter : int
            Number of iteration per chain

        - n_samples: int
            Number of retained samples per chain

        - outputdir : str
            Full path specifying where to save MCMC samples, their summary, and
            figures (traceplots and bivariate). Before running MCMC, everything
            under this path will be removed.

        - prior_for_hyperparameter (optional) : dict, default = None
            Prior parameter values for hyper-parameters. If None (default), the
            improper non-informative prior is used. This needs to be dict of
            dicts.
                e.g., {"alpha": {"mu0": 0, "kappa0": 0,
                                 "nu0": 0, "sigma20": 0},
                        "beta": {"alpha0": 0, "beta0": 0}}
            See doc string for HyperParameter for more details.

        - startWithMLE (optional) : bool, default = True
            Whether to start a chain with maximum likelihood estimation.
            This estimation pools groups and uses Nelder-Mead method.

        - chain_seed (optional) : iterable, default = range(n_chains)
            Seed for random number generation. The length of the seeds has to
            be the same as the number of chains.

        - loglikelihood_timeout (optional) : int, default = 60
            How many seconds to wait for likelihood computation. By default, if
            the logLikelihoodFunction does not return a value within 60
            seconds, the likelihood is treated as 0. This may help speeding up
            MCMC sampling.

        - n_processes (optional) : int, default = 1
            How many processes to launch. If 1 (default), multiprocessing is
            not triggered. If zero, all the available cores are used.

        - clear_outputdir (optional) : bool, default = True
            Whether to delete all the files in the outputdir.

        - saveLogLikelihood (optional) : bool, defalt = True
            Whether to save loglikelihood when saving samples. Saved
            loglikelihood can be used to compute waic or loo with loo package
            in R.

        - verbose (optional) : bool, default = True
            Whether to display progress.

    """

    assert(n_iter >= n_samples)
    started = datetime.datetime.now()

    if verbose:
        print("Initialised at %s." % started.strftime(DATETIMEFMT))

    if clear_outputdir and os.path.exists(outputdir):
        shutil.rmtree(outputdir)

    if chain_seed is None:
        chain_seed = range(n_chains)
    assert(len(chain_seed) == n_chains)

    sampleDirectory = outputdir + "/sample/"
    os.makedirs(sampleDirectory, exist_ok=True)

    if n_iter // 2 > n_samples:
        burn = n_iter // 2
    else:
        burn = n_iter - n_samples

    thin = int(numpy.ceil((n_iter - burn) / n_samples))
    tuneInterval = 100

    if n_processes == 0:
        n_processes = min(n_chains, multiprocessing.cpu_count())

    if n_processes == 1:
        for chain, seed in enumerate(chain_seed):
            _sample(
                parameterName, parameter_family, startingPointValueRange,
                nGroups, nResponsesPerGroup, logLikelihoodFunction,
                prior_for_hyperparameter,
                seed, chain, n_iter, n_samples, burn, thin, tuneInterval,
                sampleDirectory, loglikelihood_timeout, startWithMLE,
                saveLogLikelihood, verbose)

    elif n_processes > 1:
        _parallel_sample(
            parameterName, parameter_family, startingPointValueRange,
            nGroups, nResponsesPerGroup, logLikelihoodFunction,
            prior_for_hyperparameter,
            chain_seed, n_iter, n_samples, burn, thin, tuneInterval,
            sampleDirectory, loglikelihood_timeout, startWithMLE, n_processes,
            saveLogLikelihood, verbose)

    else:
        print("Invalid number of processes: ", n_processes)
        print("Exiting.")
        sys.exit()

    finished = datetime.datetime.now()
    elapsed = finished - started
    print("Finished at %s. The elapsed time in total is %s." %
          (finished.strftime(DATETIMEFMT),
           datetime.timedelta(seconds=int(elapsed.total_seconds()))),
          end="\n\n")


def _sample(parameterName, parameter_family, startingPointValueRange,
            nGroups, nResponsesPerGroup, logLikelihoodFunction,
            prior_for_hyperparameter,
            seed, chain, n_iter, n_samples, burn, thin, tuneInterval,
            sampleDirectory, loglikelihood_timeout, startWithMLE,
            saveLogLikelihood, verbose):

    numpy.random.seed(seed)

    method = HierarchicalBayes(
        parameterName, parameter_family, startingPointValueRange,
        nGroups, nResponsesPerGroup, logLikelihoodFunction,
        prior_for_hyperparameter=prior_for_hyperparameter,
        loglikelihood_timeout=loglikelihood_timeout,
        startWithMLE=startWithMLE,
        verbose=verbose)

    mcmc = MCMC(
        method, chain,
        sampleDirectory=sampleDirectory,
        saveLogLikelihood=saveLogLikelihood,
        displayProgress=verbose,
        verbose=verbose)

    mcmc.sample(n_iter, burn, thin, tuneInterval)


def _parallel_sample(parameterName, parameter_family, startingPointValueRange,
                     nGroups, nResponsesPerGroup, logLikelihoodFunction,
                     prior_for_hyperparameter,
                     chain_seed, n_iter, n_samples, burn, thin, tuneInterval,
                     sampleDirectory, loglikelihood_timeout, startWithMLE,
                     n_processes, saveLogLikelihood, verbose):

    n_chains = len(chain_seed)

    processes = [Process(target=_sample,
                         args=(parameterName, parameter_family,
                               startingPointValueRange,
                               nGroups, nResponsesPerGroup,
                               logLikelihoodFunction,
                               prior_for_hyperparameter,
                               chain_seed[chain], chain, n_iter, n_samples,
                               burn, thin, tuneInterval,
                               sampleDirectory, loglikelihood_timeout,
                               startWithMLE,
                               saveLogLikelihood,
                               verbose and (chain % n_processes) == 0))
                 for chain in range(n_chains)]

    processings = []
    for i, process in enumerate(processes):
        process.start()
        processings.append(process)

        if len(processings) % n_processes == 0:
            for processing in processings:
                processing.join()
            processings.clear()
