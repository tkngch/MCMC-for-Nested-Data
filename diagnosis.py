#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import numpy
import pandas
from scipy.stats import kde
import matplotlib.pyplot as plt


def _stdout_csv(content):
    print("\t" + content.replace(",", ", ").replace("\n", "\n\t"))


def _compute_hpd_interval(samples, hdi_p):
    prob = hdi_p / 100.
    sorted_samples = numpy.array(sorted(samples))
    n_samples = len(samples)
    gap = max(1, min(n_samples - 1, round(n_samples * prob)))
    init = numpy.array(range(n_samples - gap))
    tmp = sorted_samples[init + gap] - sorted_samples[init]
    inds = numpy.where(tmp == min(tmp))[0][0]
    interval = (sorted_samples[inds], sorted_samples[inds + gap])

    return interval


class Diagnostic(object):
    def __init__(self, sampledir):
        """
        Diagnostic class.

        This class computes rhat and the effective number of samples and saves
        them in summary.csv under the same directory as sampledir.

        The computation is as defined in Gelman, A., Carlin, J., Stern, H.,
        Dunson, D., Vehtari, A., and Rubin, D. (2013). Bayesian Data Analysis,
        Third Edition.

        """

        sample_files = glob.glob(sampledir + "/sample*.csv")
        self._organise_samples(sample_files)

        self._B = None
        self._W = None
        self._vhat = None
        self._rho = None

        self._rhat = None
        self._effective_n = None
        self._median = None
        self._hdi = None
        self._hdi_p = 95
        self._summary = None

    def _organise_samples(self, sample_files):
        """

        This loads up the csv files with MCMC samples, and divide samples from
        each chain into first and second halves, as required for the
        computation.

        """

        self._m = len(sample_files) * 2
        self._samples = {}
        for i, filename in enumerate(sample_files):
            d = pandas.read_csv(filename, engine="python")

            if i == 0:
                n = d.shape[0]
                self._n = n // 2
                index = dict((key, 0) for key in d.dtypes.index)

            for key in d.dtypes.index:
                if key in ("chain", "index"):
                    continue

                if i == 0:
                    self._samples[key] = numpy.zeros((self._m, self._n))

                samples = d[key].tolist()
                self._samples[key][index[key], :] = samples[0:self._n]
                index[key] += 1
                self._samples[key][index[key], :] = samples[self._n:n]
                index[key] += 1

    def _compute_between_sequence_variance(self):
        if self._B is not None:
            return 0

        self._B = {}
        for key in self._samples:
            self._B[key] = self._n * numpy.var(numpy.mean(self._samples[key],
                                                          axis=1),
                                               ddof=1)

    def _compute_within_sequence_variance(self):
        if self._W is not None:
            return 0

        self._W = {}
        for key in self._samples:
            self._W[key] = numpy.mean(numpy.var(self._samples[key],
                                                axis=1,
                                      ddof=1))

    def _compute_marginal_posterior_variance(self):
        if self._vhat is not None:
            return 0

        self._vhat = {}
        self._compute_between_sequence_variance()
        self._compute_within_sequence_variance()
        for key in self._samples:
            self._vhat[key] = self._W[key] * (self._n - 1) / self._n + \
                              self._B[key] / self._n

    def _compute_variogram(self, t, key):
        return (sum(sum((self._samples[key][j][i] -
                         self._samples[key][j][i - t]) ** 2
                    for i in range(t, self._n))
                    for j in range(self._m)) /
                (self._m * (self._n - t)))

    def _compute_autocorrelation(self):
        if self._rho is not None:
            return 0

        self._rho = {}
        self._compute_marginal_posterior_variance()
        for key in self._samples:
            self._rho[key] = numpy.zeros(self._n)

            for t in range(self._n):
                self._rho[key][t] = 1. - \
                                    self._compute_variogram(t, key) / \
                                    (2. * self._vhat[key])

    @property
    def rhat(self):
        if self._rhat is None:
            self._compute_rhat()
        return self._rhat

    def _compute_rhat(self):
        if self._rhat is not None:
            return 0

        self._rhat = {}
        self._compute_within_sequence_variance()
        self._compute_marginal_posterior_variance()
        for key in self._samples:
            self._rhat[key] = numpy.sqrt(self._vhat[key] / self._W[key])

    @property
    def effective_n(self):
        if self._effective_n is None:
            self._compute_effective_n()
        return self._effective_n

    def _compute_effective_n(self):
        if self._effective_n is not None:
            return 0

        self._effective_n = {}
        self._compute_marginal_posterior_variance()
        self._compute_autocorrelation()
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

            self._effective_n[key] = (self._m * self._n) /\
                                     (1 + 2 *
                                      numpy.sum(self._rho[key][0:T + 1]))

    @property
    def summary(self):
        if self._summary is None:
            self._summarise()
        return self._summary

    def _summarise(self):
        if self._summary is not None:
            return 0

        self._compute_rhat()
        self._compute_effective_n()
        self._compute_median_and_hdi()

        self._summary = numpy.array([(key.encode(),
                                      self._rhat[key], self._rhat[key] < 1.1,
                                      self._effective_n[key],
                                      self._effective_n[key] > self._m * 10,
                                      self._median[key],
                                      self._hdi[key][0],
                                      self._hdi[key][1])
                                     for key in self._samples],
                                    dtype=[("parameter", "S40"),
                                           ("rhat", float),
                                           ("converged", bool),
                                           ("effective_n", float),
                                           ("enough n", bool),
                                           ("median", float),
                                           ("HDI lower", float),
                                           ("HDI upper", float)])
        self._summary = numpy.sort(self._summary, order="parameter")

    def print(self, csvfile=None, hyperonly=False):
        """
        Print out summary onto either csvfile or stdout.

        :Arguments:

            - csvfile (optional) : str, default = None
                The file name with the path to store the sample summary. When
                None, the summary is printed on stdout.

            - hyperonly (optional) : bool, default = False
                When True, the summary for only the hyper-parameters is
                printed.

        """
        self._summarise()

        if csvfile is None:
            print("MCMC convergence diagnostic.")

        out = ",".join([key for key in self._summary.dtype.names])
        out += "\n"
        for row in self._summary:
            if hyperonly and (b"_" not in row[0]):
                continue

            out += "'%s',%.3f,%s,%.3f,%s,%.3f,%.3f,%.3f\n" % \
                   (row[0].decode("ascii"), row[1], row[2], row[3],
                    row[4], row[5], row[6], row[7])

        if csvfile is None:
            _stdout_csv(out)

        else:
            with open(csvfile, "w") as h:
                h.write(out)

    @property
    def median(self):
        if self._median is None:
            self._compute_median_and_hdi()
        return self._median

    @property
    def hdi(self):
        if self._hdi is None:
            self._compute_median_and_hdi()
        return self._hdi

    def _compute_median_and_hdi(self):
        if self._median is not None and self._hdi is not None:
            return 0

        self._median, self._hdi = {}, {}
        for key in self._samples:
            samples = self._samples[key].flatten()
            self._median[key] = numpy.median(samples)
            self._hdi[key] = _compute_hpd_interval(samples, self._hdi_p)


class Summary(object):
    """
    Summarise individual parameter values.
    """
    def __init__(self, sampledir):
        """
        :Arguments:

            - sampledir : str
                A path to the directory where sample csv files are stored.

        """
        self._load_samples(sampledir)
        self._summarise()

    def _load_samples(self, sampledir):
        sample_files = glob.glob(sampledir + "/sample*.csv")

        self._mean = {}
        self._median = {}

        for i, filename in enumerate(sample_files):
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
        self._summary = "stats,parameter,mean,HDI lower,HDI upper\n"

        for s, d in zip(("mean", "median"), (self._mean, self._median)):
            for name in sorted(d):
                m = numpy.mean(d[name])
                hdi = _compute_hpd_interval(d[name], 95.)

                self._summary += "%s,%s,%.4f,%.4f,%.4f\n"\
                    % (s, name, m, hdi[0], hdi[1])

    def print(self, csvfile=None):
        if csvfile is None:
            print("Summary of individual parameters")
            _stdout_csv(self._summary)
        else:
            with open(csvfile, "w") as h:
                h.write(self._summary)


class Figure(object):
    def __init__(self, sampledir):
        """
        Figure class to assess mixing and convergence of MCMC chains.

        :Arguments:

            - sampledir : str
                A path to the directory where sample csv files are stored.

        """

        plt.close("all")

        self._load_samples(sampledir)
        self._load_summary(sampledir)
        self._load_loglikelihoods(sampledir)

        suffices = numpy.unique(["[" + name.split("[")[1]
                                 for name in self._keys if "[" in name])
        self._key_suffices = numpy.hstack([["_"], suffices])

        self._colours = numpy.array(["blue", "red", "green", "magenta",
                                     "cyan", "yellow", "black"])[:self._m]

    def _load_samples(self, sampledir):
        sample_files = glob.glob(sampledir + "/sample*.csv")

        self._m = len(sample_files)
        self._samples = {}
        self._value_range = {}

        for i, filename in enumerate(sample_files):
            d = pandas.read_csv(filename, engine="python")

            if i == 0:
                self._n = d.shape[0]
                self._keys = [key for key in d.dtypes.index
                              if key not in ("chain", "index")]

            for key in self._keys:
                if i == 0:
                    self._samples[key] = numpy.zeros((self._n, self._m))

                self._samples[key][:, i] = d[key].tolist()

    def _load_summary(self, sampledir):
        self._summary = pandas.read_csv(sampledir + "/summary.csv",
                                        quotechar="'")

    def _load_loglikelihoods(self, sampledir):
        loglikelihood_files = glob.glob(sampledir + "/log_likelihood*.csv")

        if len(loglikelihood_files) == 0:
            self._loglikelihoods = None
            return

        size = sum([os.path.getsize(filename)
                    for filename in loglikelihood_files])
        if size == 0:
            self._loglikelihoods = None
            return

        # engine=python. otherwise read_csv segfaults for some files.
        loglikelihoods = [pandas.read_csv(
                            filename, header=None, engine="python"
                          ).sum(axis=1)
                          for filename in loglikelihood_files]
        self._loglikelihoods = pandas.concat(loglikelihoods, axis=1)

    def loglikelihood(self, dest=None):
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

    def traceplots(self, dest, n=30):
        """
        Creates traceplots. Useful for assessing mixing and convergence.

        :Arguments:

            - dest : str
                A path to directory to save figures

            - n (optional) : int, default = 30
                How many traceplots to create.

        """

        for i, key_suffix in enumerate(self._key_suffices[:n]):

            progress = "Creating traceplots: %i out of %i" % (i + 1, n)
            if i != 0:
                print("\r" * len(progress), end="")
            print(progress, end="")

            plotname = dest + "/traceplot%s.png" % key_suffix
            keys = sorted([key for key in self._keys if key_suffix in key])
            self.traceplot(keys, plotname)

        print("\r" * len(progress), end="")
        print(" " * len(progress), end="")
        print("\r" * len(progress), end="")
        print("Creating traceplots: Done.")

    def traceplot(self, keys=None, dest=None):
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
            rhat = self._summary[self._summary["parameter"] == key]["rhat"]

            title = key.replace("_", " ")
            title += " (rhat=%.3f)" % float(rhat)

            self._kdeplot(ax[i, 0], key)
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

    def _kdeplot(self, ax, key):
        for i in range(self._m):
            d = self._samples[key][:, i]

            if numpy.var(d) > 1e-8:
                density = kde.gaussian_kde(d)
                l = numpy.min(d)
                u = numpy.max(d)
                x = numpy.linspace(0, 1, 100) * (u - l) + l
                ax.plot(x, numpy.log(density(x)),
                        color=self._colours[i])

    def _plot(self, ax, key):
        for i in range(self._m):
            ax.plot(self._samples[key][:, i], color=self._colours[i])

    def bivariates(self, dest, n=30):
        """

        Creates bivariate plots of samples. Useful for assessing pairwise
        relationship between variables.

        :Arguments:

            - dest : str
                A path to directory to save figures

            - n (optional) : int, default = 30
                How many plots to create.

        """

        for i, key_suffix in enumerate(self._key_suffices[:n]):

            progress = "Creating bivariate plots: %i out of %i" % (i + 1, n)
            if i != 0:
                print("\r" * len(progress), end="")
            print(progress, end="")

            plotname = dest + "/bivariate%s.png" % key_suffix
            keys = sorted([key for key in self._keys if key_suffix in key])
            self.bivariate(keys, plotname)

        print("\r" * len(progress), end="")
        print(" " * len(progress), end="")
        print("\r" * len(progress), end="")
        print("Creating bivariate plots: Done.")

    def bivariate(self, keys=None, dest=None):
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


def diagnose(outputdir):
    sampledir = outputdir + "/sample/"
    traceplotdir = outputdir + "/figure/traceplot/"
    bivariatedir = outputdir + "/figure/bivariate/"
    ll_file = outputdir + "/figure/log_likelihood.png"

    for directory in (traceplotdir, bivariatedir):
        os.makedirs(directory, exist_ok=True)

    diagnostic = Diagnostic(sampledir)
    diagnostic.print(sampledir + "/summary.csv")
    diagnostic.print(sampledir + "/summary_hyperonly.csv", hyperonly=True)
    diagnostic.print(hyperonly=True)

    summary = Summary(sampledir)
    summary.print(sampledir + "/individual_summary.csv")
    summary.print()

    fig = Figure(sampledir)
    fig.loglikelihood(ll_file)
    fig.traceplots(traceplotdir)
    fig.bivariates(bivariatedir)
