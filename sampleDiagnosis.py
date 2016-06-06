#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import numpy
import pandas
import matplotlib.pyplot as plt


def diagnoseSamples(outputDirectory,
                    assessConvergence=True,
                    printSummary=True,
                    nFigures=10):
    """
    Diagnose samples.

    :Arguments:

        - outputDirectory : str
            Full path specifying where MCMC samples are saved. This path should
            be the same as what you give samplePosterior in posteriorSampling
            module.

        - assessConvergence (optional) : bool, default = True
            Whether to compute rhat, effective number of samples, etc to assess
            convergence of MCMC chains.

        - printSummary (optional) : bool, default = True
            Whether to calculate summary statistics of samples. Summary
            statistics are computed with parameter values, ignoring
            hyper-parameters.

            So groupMean is the mean of parameters for each group, which may or
            may not be the same as the group mean estimated with
            hyper-parameter. So with partial-pooling, this summary may not be
            useful. Though with no-pooling, this summary statistics are
            considered to be group estimates.

        - nFigures (optional) : int, default = 10
            How many trace plots and bivariate plots to make. Useful in
            assessing convergence.

    """

    sampleDirectory = outputDirectory + "/sample/"
    diagnosticDirectory = outputDirectory + "/diagnostic/"
    os.makedirs(diagnosticDirectory, exist_ok=True)

    if assessConvergence:
        print("- Convergence Diagnostic -")
        diagnostic = Diagnostic(sampleDirectory)
        path = diagnosticDirectory + "/diagnosticAssessment.csv"
        diagnostic.print(path, False, False)

        if diagnostic.completelyPooled:
            diagnostic.print(None, False, False)

        if diagnostic.partiallyPooled:
            path = diagnosticDirectory + "/diagnosticAssessmentHyperOnly.csv"
            diagnostic.print(path, False, True)
            diagnostic.print(None, False, True)

        if not diagnostic.completelyPooled:
            path = diagnosticDirectory + "/diagnosticAssessmentIndividual.csv"
            diagnostic.print(path, True, False)
            diagnostic.print(None, True, False)

    if printSummary:
        summary = Summary(sampleDirectory)
        summary.print(sampleDirectory + "/summary.csv")
        summary.print(None)

    traceplotDirectory = outputDirectory + "/figure/traceplot/"
    bivariateDirectory = outputDirectory + "/figure/bivariate/"
    llFile = outputDirectory + "/figure/logLikelihood.png"

    if nFigures > 0:
        for directory in (traceplotDirectory, bivariateDirectory):
            os.makedirs(directory, exist_ok=True)

        fig = Figure(sampleDirectory)
        fig.loglikelihood(llFile)
        fig.traceplots(traceplotDirectory, nFigures)
        fig.bivariates(bivariateDirectory, nFigures)


class Diagnostic(object):
    def __init__(self, sampleDirectory):
        """
        Diagnostic class.

        This class computes rhat and the effective number of samples and saves
        them in summary.csv under the same directory as sampleDirectory.

        The computation is as defined in Gelman, A., Carlin, J., Stern, H.,
        Dunson, D., Vehtari, A., and Rubin, D. (2013). Bayesian Data Analysis,
        Third Edition.

        """

        sampleFiles = glob.glob(sampleDirectory + "/sample*.csv")
        self._organiseSamples(sampleFiles)

        self._B = None
        self._W = None
        self._vhat = None
        self._rho = None

        self._rhat = None
        self._effectiveN = None
        self._median = None
        self._hdi = None
        self._hdiP = 95
        self._assessment = None
        self._summary = None

    def _organiseSamples(self, sampleFiles):
        """

        This loads up the csv files with MCMC samples, and divide samples from
        each chain into first and second halves, as required for the
        computation.

        """

        self._m = len(sampleFiles) * 2
        self._samples = {}
        self.partiallyPooled = False
        self.completelyPooled = True
        for i, filename in enumerate(sampleFiles):
            d = pandas.read_csv(filename, engine="python")

            if i == 0:
                n = d.shape[0]
                self._n = n // 2
                index = dict((key, 0) for key in d.dtypes.index)

            for key in d.dtypes.index:
                if key in ("chain", "index"):
                    continue

                if "_" in key:
                    self.partiallyPooled = True

                if "01]" in key:
                    self.completelyPooled = False

                if i == 0:
                    self._samples[key] = numpy.zeros((self._m, self._n))

                samples = d[key].tolist()
                self._samples[key][index[key], :] = samples[0:self._n]
                index[key] += 1
                self._samples[key][index[key], :] = samples[self._n:n]
                index[key] += 1

    def _computeBetweenSequenceVariance(self):
        if self._B is not None:
            return

        self._B = {}
        for key in self._samples:
            self._B[key] = self._n * numpy.var(numpy.mean(self._samples[key],
                                                          axis=1),
                                               ddof=1)

    def _computeWithinSequenceVariance(self):
        if self._W is not None:
            return

        self._W = {}
        for key in self._samples:
            self._W[key] = numpy.mean(numpy.var(self._samples[key],
                                                axis=1,
                                      ddof=1))

    def _computeMarginalPosteriorVariance(self):
        if self._vhat is not None:
            return

        self._vhat = {}
        self._computeBetweenSequenceVariance()
        self._computeWithinSequenceVariance()
        for key in self._samples:
            self._vhat[key] = self._W[key] * (self._n - 1) / self._n + \
                              self._B[key] / self._n

    def _computeVariogram(self, t, key):
        return (sum(sum((self._samples[key][j][i] -
                         self._samples[key][j][i - t]) ** 2
                    for i in range(t, self._n))
                    for j in range(self._m)) /
                (self._m * (self._n - t)))

    def _computeAutocorrelation(self):
        if self._rho is not None:
            return

        self._rho = {}
        self._computeMarginalPosteriorVariance()
        for key in self._samples:
            self._rho[key] = numpy.zeros(self._n)

            for t in range(self._n):
                self._rho[key][t] = 1. - \
                                    self._computeVariogram(t, key) / \
                                    (2. * self._vhat[key])

    @property
    def rhat(self):
        if self._rhat is None:
            self._computeRhat()
        return self._rhat

    def _computeRhat(self):
        if self._rhat is not None:
            return

        self._rhat = {}
        self._computeWithinSequenceVariance()
        self._computeMarginalPosteriorVariance()
        for key in self._samples:
            self._rhat[key] = numpy.sqrt(self._vhat[key] / self._W[key])

    @property
    def effectiveN(self):
        if self._effectiveN is None:
            self._computeEffectiveN()
        return self._effectiveN

    def _computeEffectiveN(self):
        if self._effectiveN is not None:
            return

        self._effectiveN = {}
        self._computeMarginalPosteriorVariance()
        self._computeAutocorrelation()
        for key in self._samples:

            fnd = False
            T = None
            for t in range(self._n - 2):
                if not fnd and not t % 2:
                    fnd = (self._rho[key][t + 1] + self._rho[key][t + 2]) < 0
                if fnd:
                    T = t
                    break

            if T is None:
                T = self._n - 1

            self._effectiveN[key] = \
                (self._m * self._n) /\
                (1 + 2 * numpy.sum(self._rho[key][0:T + 1]))

    @property
    def assessment(self):
        if self._assessment is None:
            self._assess()
        return self._assessment

    def _assess(self):
        if self._assessment is not None:
            return

        self._computeRhat()
        self._computeEffectiveN()
        self._computeMedianAndHdi()

        self._assessment =\
            numpy.array([(key.encode(),
                          self._rhat[key],
                          self._rhat[key] < 1.1,
                          self._effectiveN[key],
                          self._effectiveN[key] > self._m * 10,
                          self._median[key],
                          self._hdi[key][0],
                          self._hdi[key][1])
                         for key in self._samples],
                        dtype=[("parameter", "S40"),
                               ("rhat", float),
                               ("converged", bool),
                               ("effective n", float),
                               ("enough n", bool),
                               ("median", float),
                               ("HDI lower", float),
                               ("HDI upper", float)])
        self._assessment = numpy.sort(self._assessment, order="parameter")

    @property
    def summary(self):
        if self._summary is None:
            self._summarise()
        return self._summary

    def _summarise(self):
        if self._summary is not None:
            return

        if self._assessment is None:
            self._assess()

        parameterNames = [name.decode("ascii").split("[")[0]
                          for name in self._assessment["parameter"]
                          if b"[" in name]
        parameterNames = sorted(list(set(parameterNames)))
        rhats = dict((name, []) for name in parameterNames)
        converged = dict((name, []) for name in parameterNames)

        for row in self._assessment:
            if b"[" not in row[0]:
                continue

            name = row[0].decode("ascii").split("[")[0]
            rhats[name].append(row[1])
            converged[name].append(row[2])

        self._summary = numpy.array([(name,
                                      min(rhats[name]),
                                      numpy.median(rhats[name]),
                                      max(rhats[name]),
                                      numpy.mean(converged[name]))
                                     for name in parameterNames],
                                    dtype=[("parameter", "S40"),
                                           ("rhat min", float),
                                           ("rhat median", float),
                                           ("rhat max", float),
                                           ("proportion converged", float)])

    def print(self, csvfile, individualSummary, hyperOnly):
        """
        Print out summary onto either csvfile or stdout.

        :Arguments:

            - csvfile : str
                The file name with the path to store the sample summary. When
                None, the summary is printed on stdout.

            - individualSummary : bool
                Whether to print summary statistics (e.g., rhat, converged)
                aggregated across groups.

            - hyperOnly : bool
                When True and samples are from partially pooled MCMC, the
                summary for only the hyper-parameters is printed.

        """
        if individualSummary and self.completelyPooled:
            raise ValueError(
                "MCMC was completely pooled. There is no individual summary.")

        if hyperOnly and not self.partiallyPooled:
            raise ValueError(
                "MCMC was not partially pooled. There is no hyper-parameter.")

        if individualSummary and hyperOnly:
            raise ValueError(
                "Choose individualSummary or hyperOnly. Not both.")

        if (csvfile is None) and hyperOnly:
            print("MCMC convergence diagnostic for hyper-parameters.")
        elif (csvfile is None) and individualSummary:
            print("Summary of MCMC convergence diagnostic.")
        elif csvfile is None:
            print("MCMC convergence diagnostic.")

        if not individualSummary:
            out = self._getAssessmentString(hyperOnly)
        else:
            out = self._getSummaryString()

        if csvfile is None:
            _stdout_csv(out)

        else:
            with open(csvfile, "w") as h:
                h.write(out)

    def _getAssessmentString(self, hyperOnly):
        self._assess()

        out = ",".join([key for key in self._assessment.dtype.names])
        out += "\n"
        for row in self._assessment:
            if hyperOnly and (b"_" not in row[0]):
                continue

            out += "'%s',%.3f,%s,%.3f,%s,%.3f,%.3f,%.3f\n" % \
                   (row[0].decode("ascii"), row[1], row[2], row[3],
                    row[4], row[5], row[6], row[7])

        return out

    def _getSummaryString(self):
        self._summarise()

        out = ",".join([key for key in self._summary.dtype.names])
        out += "\n"
        for row in self._summary:
            out += "'%s',%.3f,%.3f,%.3f,%.3f\n" % \
                   (row[0].decode("ascii"), row[1], row[2], row[3], row[4])

        return out

    @property
    def median(self):
        if self._median is None:
            self._computeMedianAndHdi()
        return self._median

    @property
    def hdi(self):
        if self._hdi is None:
            self._computeMedianAndHdi()
        return self._hdi

    def _computeMedianAndHdi(self):
        if self._median is not None and self._hdi is not None:
            return

        self._median, self._hdi = {}, {}
        for key in self._samples:
            samples = self._samples[key].flatten()
            self._median[key] = numpy.median(samples)
            self._hdi[key] = _computeHpdInterval(samples, self._hdiP)


class Summary(object):
    """
    Summarise individual parameter values.
    """
    def __init__(self, sampleDirectory):
        """
        :Arguments:

            - sampleDirectory : str
                A path to the directory where sample csv files are stored.

        """
        self._loadSamples(sampleDirectory)
        self._summarise()

    def _loadSamples(self, sampleDirectory):
        sampleFiles = glob.glob(sampleDirectory + "/sample*.csv")

        self._mean = {}
        self._median = {}

        for i, filename in enumerate(sampleFiles):
            d = pandas.read_csv(filename, engine="python")

            if i == 0:
                self._n = d.shape[0]
                self._names = numpy.unique([name.split("[")[0]
                                            for name in d.dtypes.index
                                            if "[" in name])

                for name in self._names:
                    self._mean[name] = []
                    self._median[name] = []

            for j in range(self._n):
                for name in self._names:
                    x = [d[key][j]
                         for key in d.dtypes.index if (name + "[") in key]

                    self._mean[name].append(numpy.mean(x))
                    self._median[name].append(numpy.median(x))

    def _summarise(self):
        self._summary = "stats,parameter,mean,median,HDI lower,HDI upper\n"

        for s, d in zip(("groupMean", "groupMedian"),
                        (self._mean, self._median)):
            for name in sorted(d):
                mean = numpy.mean(d[name])
                median = numpy.median(d[name])
                hdi = _computeHpdInterval(d[name], 95.)

                self._summary += "%s,%s,%.4f,%.4f,%.4f,%.4f\n"\
                    % (s, name, mean, median, hdi[0], hdi[1])

    def print(self, csvfile):
        if csvfile is None:
            print("Summary of individual parameters.")
            _stdout_csv(self._summary)
        else:
            with open(csvfile, "w") as h:
                h.write(self._summary)


class Figure(object):
    def __init__(self, sampleDirectory):
        """
        Figure class to assess mixing and convergence of MCMC chains.

        :Arguments:

            - sampleDirectory : str
                A path to the directory where sample csv files are stored.

        """

        plt.close("all")

        self._loadSamples(sampleDirectory)
        self._loadSummary(sampleDirectory)
        self._loadLogLikelihoods(sampleDirectory)

        suffices = numpy.unique(["[" + name.split("[")[1]
                                 for name in self._keys if "[" in name])

        partiallyPooled = any(["_" in name for name in self._keys])
        if partiallyPooled:
            self._keySuffices = numpy.hstack([["_"], suffices])
        else:
            self._keySuffices = suffices

        self._colours = numpy.array(["blue", "red", "green", "magenta",
                                     "cyan", "yellow", "black"])[:self._m]

    def _loadSamples(self, sampleDirectory):
        sampleFiles = glob.glob(sampleDirectory + "/sample*.csv")

        self._m = len(sampleFiles)
        self._samples = {}
        self._value_range = {}

        for i, filename in enumerate(sampleFiles):
            d = pandas.read_csv(filename, engine="python")

            if i == 0:
                self._n = d.shape[0]
                self._keys = [key for key in d.dtypes.index
                              if key not in ("chain", "index")]

            for key in self._keys:
                if i == 0:
                    self._samples[key] = numpy.zeros((self._n, self._m))

                self._samples[key][:, i] = d[key].tolist()

    def _loadSummary(self, sampleDirectory):
        path = sampleDirectory + "../diagnostic/diagnosticAssessment.csv"
        if os.path.exists(path):
            self._diagnosticAssessment = pandas.read_csv(path, quotechar="'")
        else:
            self._diagnosticAssessment = None

    def _loadLogLikelihoods(self, sampleDirectory):
        logLikelihoodFiles = glob.glob(sampleDirectory + "/logLikelihood*.csv")

        if len(logLikelihoodFiles) == 0:
            self._loglikelihoods = None
            return

        size = sum([os.path.getsize(filename)
                    for filename in logLikelihoodFiles])
        if size == 0:
            self._loglikelihoods = None
            return

        # engine=python. otherwise read_csv segfaults for some files.
        loglikelihoods = [pandas.read_csv(
                            filename, header=None, engine="python"
                          ).sum(axis=1)
                          for filename in logLikelihoodFiles]
        self._loglikelihoods = pandas.concat(loglikelihoods, axis=1)

    def loglikelihood(self, dest):
        if self._loglikelihoods is None:
            return
        print("Creating loglikelihood plot", end="")

        plt.figure(figsize=(12, 3))

        for i in range(self._m):
            plt.plot(self._loglikelihoods[i], color=self._colours[i], label=i)

        plt.ylabel("Log Likelihood")
        plt.xlabel("Iteration")
        plt.legend(title="Chain")

        plt.tight_layout()
        if dest is None:
            plt.show()
        else:
            plt.savefig(dest)
            plt.close()

        print(": Done")

    def traceplots(self, dest, n):
        """
        Creates traceplots. Useful for assessing mixing and convergence.

        :Arguments:

            - dest : str
                A path to directory to save figures

            - n (optional) : int, default = 30
                How many traceplots to create.

        """

        for i, keySuffix in enumerate(self._keySuffices[:n]):

            progress = "Creating traceplots: %i out of %i" % (i + 1, n)
            if i != 0:
                print("\r" * len(progress), end="")
            print(progress, end="")

            plotname = dest + "/traceplot%s.png" % keySuffix
            keys = sorted([key for key in self._keys if keySuffix in key])
            self.traceplot(keys, plotname)

        print("\r" * len(progress), end="")
        print(" " * len(progress), end="")
        print("\r" * len(progress), end="")
        print("Creating traceplots: Done.")

    def traceplot(self, keys, dest):
        """
        Creates one traceplot.
        """

        if keys is None:
            keys = sorted([key for key in self._keys if "_mean" in key])

        n = len(keys)
        figsize = (12, n * 2)
        fig, ax = plt.subplots(n, 2, figsize=figsize)

        if n == 1:
            ax = numpy.matrix([ax[0], ax[1]])

        for i, key in enumerate(keys):
            title = key.replace("_", " ")

            if self._diagnosticAssessment is not None:
                index = self._diagnosticAssessment["parameter"] == key
                rhat = self._diagnosticAssessment[index]["rhat"]
                title += " (rhat=%.3f)" % float(rhat)

            self._hist(ax[i, 0], key)
            ax[i, 0].set_title(title)
            ax[i, 0].set_xlabel("Sample Value")
            ax[i, 0].set_ylabel("Log Density")
            ax[i, 0].set_yticks([])

            self._plot(ax[i, 1], key)
            ax[i, 1].set_title(title)
            ax[i, 1].set_xlabel("Iteration")
            ax[i, 1].set_ylabel("Sample Value")

        plt.tight_layout()
        if dest is None:
            plt.show()
        else:
            plt.savefig(dest)
            plt.close()

    # def _kdeplot(self, ax, key):
    #     for i in range(self._m):
    #         d = self._samples[key][:, i]
    #
    #         if numpy.var(d) > 1e-8:
    #             density = kde.gaussian_kde(d)
    #             l = numpy.min(d)
    #             u = numpy.max(d)
    #             x = numpy.linspace(0, 1, 100) * (u - l) + l
    #             ax.plot(x, numpy.log(density(x)),
    #                     color=self._colours[i])

    def _hist(self, ax, key):
        for i in range(self._m):
            d = self._samples[key][:, i]

            ax.hist(d, bins=len(d) // 10,
                    histtype="stepfilled", alpha=0.5,
                    color=self._colours[i], normed=True)

    def _plot(self, ax, key):
        for i in range(self._m):
            ax.plot(self._samples[key][:, i], color=self._colours[i])

    def bivariates(self, dest, n):
        """

        Creates bivariate plots of samples. Useful for assessing pairwise
        relationship between variables.

        :Arguments:

            - dest : str
                A path to directory to save figures

            - n (optional) : int, default = 30
                How many plots to create.

        """

        for i, keySuffix in enumerate(self._keySuffices[:n]):

            progress = "Creating bivariate plots: %i out of %i" % (i + 1, n)
            if i != 0:
                print("\r" * len(progress), end="")
            print(progress, end="")

            plotname = dest + "/bivariate%s.png" % keySuffix
            keys = sorted([key for key in self._keys if keySuffix in key])
            self.bivariate(keys, plotname)

        print("\r" * len(progress), end="")
        print(" " * len(progress), end="")
        print("\r" * len(progress), end="")
        print("Creating bivariate plots: Done.")

    def bivariate(self, keys, dest):
        """
        Create one bivariate plot.
        """

        if keys is None:
            keys = sorted([key for key in self._keys if "_mean" in key])

        n = len(keys)
        if n == 1:
            return 0

        figsize = (n * 2, n * 2)
        fig, ax = plt.subplots(n, n, figsize=figsize)

        for i, keyy in enumerate(keys):
            for j, keyx in enumerate(keys):

                if i == j:
                    ax[i, j].text(0.5, 0.5, keyx.replace("_", " "),
                                  horizontalalignment="center",
                                  verticalalignment="center",
                                  transform=ax[i, j].transAxes)
                    ax[i, j].set_axis_off()
                    continue

                for k in range(self._m):
                    ax[i, j].scatter(self._samples[keyx][:, k],
                                     self._samples[keyy][:, k],
                                     color=self._colours[k],
                                     s=5, alpha=0.1)

        plt.tight_layout()
        if dest is None:
            plt.show()
        else:
            plt.savefig(dest)
            plt.close()


def _stdout_csv(content):
    print("\t" + content.replace(",", ", ").replace("\n", "\n\t"))


def _computeHpdInterval(samples, hdi_p):
    prob = hdi_p / 100.
    sorted_samples = numpy.array(sorted(samples))
    n_samples = len(samples)
    gap = max(1, min(n_samples - 1, round(n_samples * prob)))
    init = numpy.array(range(n_samples - gap))
    tmp = sorted_samples[init + gap] - sorted_samples[init]
    inds = numpy.where(tmp == min(tmp))[0][0]
    interval = (sorted_samples[inds], sorted_samples[inds + gap])

    return interval
