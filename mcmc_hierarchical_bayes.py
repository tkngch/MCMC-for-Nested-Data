#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015 Takao Noguchi (tkngch@runbox.com)

"""

MCMC with Hierarchical Bayes

This file declares and defines the classes and the functions to sample a
posterior distribution using MCMC algorithm.

"""

import signal
import glob
import shutil
import os.path
import datetime
from multiprocessing import Process
import numpy
import scipy.stats
import scipy.optimize
import pandas
from scipy.stats import kde
import matplotlib.pyplot as plt


# ---------------- #
# global variables #
# ---------------- #
NEGATIVE_INFINITY = -1e300
DATETIMEFMT = "%d-%m-%Y %H:%M"


# ---------------------------- #
# class declaration/definition #
# ---------------------------- #

class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError()


class Parameter(object):
    def __init__(self, parameter_name, index, family, prior,
                 proposal_sd=1., verbose=False):
        """

        Parameter class defines, well, parameter. This class is automatically
        created and used in HierarchicalBayes class. Users are not expected to
        use this class.

        :Arguments:

            - parameter_name : string

            - index : int
                This is used as a group index, and the unique name of the
                parameter is set to "%s[%.3i]" % (parameter_name, index).

            - family : str
                "gaussian", "log normal", "negated log normal",
                "exponential", "negated exponential",
                "poisson", or "negated poisson".

                See docstring for HyperParameter.

            - prior : scipy.stats distribution

            - proposal_sd (optional) : double, default = 1.

            - verbose (optional) : bool, default = False

        """

        self._verbose = verbose or False
        self._parameter_name = parameter_name
        self._unique_name = "%s[%.3i]" % (parameter_name, index)
        self._family = family
        self._value = None
        self._log_likelihood = numpy.nan
        self.set_prior(prior)

        self._proposal_sd = proposal_sd
        self._adaptive_scale_factor = 1.
        # self._adaptive_scale_factor = self._value * 0.1
        self._n_accepted = 0.
        self._n_rejected = 0.

    def set_prior(self, prior):
        self._prior = prior

        if self._value is None:
            self._value = self._sample_prior()

        self._update_logprior()
        self._update_logp()

    def _sample_prior(self):
        if "log normal" in self._family:
            value = numpy.exp(self._prior.rvs())
        else:
            value = self._prior.rvs()

        if "negated" in self._family:
            value *= -1

        return value

    def _update_logprior(self):
        self._log_prior = self._get_logprior(self._value)

    def _get_logprior(self, value):
        if "negated" in self._family:
            value = -1 * self._value

        if "log normal" in self._family:
            logprior = self._prior.logpdf(numpy.log(value))
        elif "poisson" in self._family:
            logprior = self._prior.logpmf(value)
        else:
            logprior = self._prior.logpdf(value)

        return logprior

    @property
    def header(self):
        return self._unique_name

    @property
    def value(self):
        return self._value

    def propose(self):
        sd = self._proposal_sd * self._adaptive_scale_factor
        self._proposal = numpy.random.normal(self._value, sd)

        if "poisson" in self._family:
            self._proposal = int(round(self._proposal))

    @property
    def proposal(self):
        return self._proposal

    @property
    def logprior(self):
        return self._log_prior

    @property
    def logp(self):
        return self._logp

    @property
    def log_likelihood(self):
        return self._log_likelihood

    @log_likelihood.setter
    def log_likelihood(self, val):
        if self._verbose:
            print("%s ll updated from %f to %f"
                  % (self._unique_name, self._log_likelihood, val))
        self._log_likelihood = val
        self._update_logp()

    def _update_logp(self):
        self._logp = self._log_prior + self._log_likelihood

    def step(self, proposal_log_likelihood):
        proposal_logprior = self._get_logprior(self._proposal)
        proposal_logp = proposal_logprior + proposal_log_likelihood

        if self._verbose > 0:
            print("%s\t" % self._unique_name, end="")
            print("value: %.3f, logll: %.3f, logp: %.3f"
                  % (self._value, self._log_likelihood, self._logp),
                  end="\t")
            print("proposal: %.3f, logll: %.3f, logp: %.3f"
                  % (self._proposal, proposal_log_likelihood, proposal_logp),
                  end="\t")

        if (not numpy.isfinite(self._logp)) and numpy.isfinite(proposal_logp):
            self._accept(proposal_log_likelihood)
            accepted = True

        elif numpy.log(numpy.random.random()) < proposal_logp - self._logp:
            self._accept(proposal_log_likelihood)
            accepted = True

        else:
            self._reject()
            accepted = False

        self._proposal = None
        return accepted

    def _accept(self, proposal_log_likelihood):
        self._value = self._proposal
        self._log_likelihood = proposal_log_likelihood

        self._update_logprior()
        self._update_logp()

        self._n_accepted += 1.
        if self._verbose > 0:
            print("accepted")

    def _reject(self):
        self._n_rejected += 1.
        if self._verbose > 0:
            print("rejected")

    def tune(self):
        """
        Tunes the scaling parameter for the proposal distribution
        according to the acceptance rate
        """

        # Flag for tuning state
        tuning = True

        # Calculate recent acceptance rate
        if not (self._n_accepted + self._n_rejected):
            return tuning
        acc_rate = self._n_accepted / (self._n_accepted + self._n_rejected)

        current_factor = self._adaptive_scale_factor

        # Switch statement
        if acc_rate < 0.001:
            # reduce by 90 percent
            self._adaptive_scale_factor *= 0.1
        elif acc_rate < 0.05:
            # reduce by 50 percent
            self._adaptive_scale_factor *= 0.5
        elif acc_rate < 0.2:
            # reduce by ten percent
            self._adaptive_scale_factor *= 0.9
        elif acc_rate > 0.95:
            # increase by factor of ten
            self._adaptive_scale_factor *= 10.0
        elif acc_rate > 0.75:
            # increase by double
            self._adaptive_scale_factor *= 2.0
        elif acc_rate > 0.5:
            # increase by ten percent
            self._adaptive_scale_factor *= 1.1
        else:
            tuning = False

        # reset rejection count
        self._n_accepted = 0.
        self._n_rejected = 0.

        # Prevent from tuning to zero
        if not self._adaptive_scale_factor:
            self._adaptive_scale_factor = current_factor
            return False

        if self._verbose:
            print("%s\t" % self._unique_name, end="")
            print("acceptance rate: %.2f\t" % acc_rate, end="")
            print("adaptive scale factor: %.2f" % self._adaptive_scale_factor)

        return tuning


class HyperParameter(object):
    def __init__(self, parameter_name, family,
                 start=None, verbose=False):
        """

        This class defines hyper-parameter. As with Parameter class, users are
        not expected to directly interact with this class.

        :Arguments:
            - parameter_name : str

            - family : str
                "gaussian", "log normal", "negated log normal",
                "exponential", "negated exponential",
                "poisson", or "negated poisson".

                This family defines how parameter values are modeled. With the
                gaussian family, for example, parameter values are assumed to
                be normally distributed, and its mean and variance become
                hyper-parameters.

                Naturally, this family also defines range of
                parameter values: with gaussian family, a parameter value can
                take any real values, but with log normal or exponential
                family, a parameter to value can only be a positive real. With
                poisson family, the parameter value can take only non-negative
                integers.

            - start (optional) : dict
                default values are:
                    {"mean": 0, "var": 1} for gaussian family
                    {"mean": 0, "var": 1} for log normal family
                    {"invrate": 1} for exponential family
                    {"rate": 1} for poisson family

            - verbose (optional) : bool, default = False

        Priors for the hyper-parameters are all non-informative improper:

            gaussian family: mean, sqrt(var) ~ unif(-inf, inf)

            log normal family: mean, sqrt(var) ~ exp(unif(-inf, inf))

            exponential family: rate ~ gamma(0, 0)

            poisson family: rate ~ gamma(1, 0)

        """

        self._parameter_name = parameter_name
        self._family = family

        if start is not None:
            self._value = start
        else:
            self._value = {
                "gaussian": {"mean": 0, "var": 1},
                "log normal": {"mean": 0, "var": 1},
                "negated log normal": {"mean": 0, "var": 1},
                "exponential": {"invrate": 1},
                "negated exponential": {"invrate": -1},
                "poisson": {"rate": 1},
                "negated poisson": {"rate": -1}
            }[self._family]

        self._hyper_parameter_name = [name for name in self._value]
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

        if "negated" in self._family:
            x = -1 * x

        if self._family == "gaussian":
            self._update_mean(x)
            self._update_var(x)

        elif "log normal" in self._family:
            self._update_mean(numpy.log(x))
            self._update_var(numpy.log(x))
            self._value["mean"] = numpy.exp(self._value["mean"])

        elif "exponential" in self._family:
            rate = self._update_rate_param(len(x), sum(x))
            self._value["invrate"] = 1. / rate

        elif "poisson" in self._family:
            # self._update_rate_param(sum(x), len(x))
            # let's say prior is gamma(1, 0)
            self._value["rate"] = self._update_rate_param(1 + sum(x), len(x))

        if "negated" in self._family:
            for key in ("mean", "invrate", "rate"):
                if key in self._value:
                    self._value[key] *= -1

        if self._verbose:
            print("\tupdated: %s" % self.values.__str__())

    def _update_mean(self, x):
        sd = numpy.sqrt(self._value["var"] / len(x))
        self._value["mean"] = numpy.random.normal(numpy.mean(x), sd)

    def _update_var(self, x):
        scale = sum([(xj - self._value["mean"]) ** 2 for xj in x]) /\
            (len(x) - 1)
        self._value["var"] = self._sample_inv_chi2(len(x) - 1, scale)

    def _sample_inv_chi2(self, v, s2):
        return scipy.stats.invgamma(v / 2., scale=(v / 2.) * s2).rvs()

    def _update_rate_param(self, shape, inv_scale):
        return scipy.stats.gamma(a=shape, scale=1./inv_scale).rvs()

    def get_distribution(self):
        if self._family == "gaussian":
            return scipy.stats.norm(loc=self._value["mean"],
                                    scale=numpy.sqrt(self._value["var"]))

        if self._family == "log normal":
            return scipy.stats.norm(loc=numpy.log(self._value["mean"]),
                                    scale=numpy.sqrt(self._value["var"]))

        if self._family == "negated log normal":
            return scipy.stats.norm(loc=numpy.log(-1 * self._value["mean"]),
                                    scale=numpy.sqrt(self._value["var"]))

        if self._family == "exponential":
            return scipy.stats.expon(scale=self._value["invrate"])

        if self._family == "negated exponential":
            return scipy.stats.expon(scale=-1 * self._value["invrate"])

        if self._family == "poisson":
            return scipy.stats.poisson(mu=self._value["rate"])

        if self._family == "negated poisson":
            return scipy.stats.poisson(mu=-1 * self._value["rate"])

    @property
    def header(self):
        return ["%s_%s" % (self._parameter_name, x)
                for x in self._hyper_parameter_name]

    @property
    def value(self):
        return [self._value[name] for name in self._hyper_parameter_name]


class HierarchicalBayes(object):
    def __init__(self,
                 parameter_name, parameter_family, parameter_value_range,
                 n_groups, n_responses_per_group,
                 loglikelihood_function,
                 loglikelihood_timeout=60,
                 start_with_mle=False, verbose=False):
        """
        This class defines the step method and finds a starting state for MCMC.

        :Arguments:

            - parameter_name : tuple of str
                e.g., ("alpha", "beta")

            - parameter_family : dict
                e.g., {"alpha": "gaussian", "beta": "exponential"}

            - parameter_value_range : dict
                e.g., {"alpha": [0, 1], "beta": [2, 5]}

                The parameter range to initialize a starting point. This is
                ignored during MCMC sampling.

            - n_groups : int

            - n_responses_per_group : int or list of int

            - loglikelihood_function : def

                A function that accepts a nested list of parameter values and
                returns a list of log likelihood.  No other arguments are
                passed to this function. If your log likelihood function
                requires any other argument, consider partial application
                (functools.partial).

                The list of parameter values is organized as:
                    [[0.7, 0.2, ..., 0.3],
                     # n_groups values of the first parameter
                     [4.3, 3.6, ..., 2.9],
                     # n_groups values of the second parameter
                     ...].

                The order of parameters is as in the parameter_name argument
                above (e.g., if parameter_name = ("alpha", "beta"), the first
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

            - loglikelihood_timeout (optional) : int, default = 60

                How many seconds to wait for likelihood computation. By
                default, if the loglikelihood_function does not return a value
                in 60 seconds, the likelihood is treated as 0. This may help
                speeding up MCMC sampling.

            - start_with_mle (optional) : bool, default = False
                Whether to start MCMC with MLE.

            - verbose (optional) : bool, default = False

        """

        self._verbose = verbose or False

        self._parameter_name = parameter_name
        self._parameter_family = parameter_family
        self._parameter_value_range = parameter_value_range

        self._n_groups = n_groups

        if type(n_responses_per_group) == int:
            self._n_responses_per_group = [n_responses_per_group] * n_groups
        elif type(n_responses_per_group) == list:
            self._n_responses_per_group = n_responses_per_group

        self._loglikelihood_function = loglikelihood_function
        self._loglikelihood_timeout = loglikelihood_timeout

        self._ll_index = numpy.hstack(
            [0, numpy.cumsum(self._n_responses_per_group)])

        self._find_starting_point()
        if start_with_mle:
            self._optimise_starting_point()

        if self._verbose:
            print("Starting Point:")

        self._parameter = {}
        for i, name in enumerate(self._parameter_name):

            family = self._parameter_family[name]
            val = self._starting_point[i]

            if family in ("gaussian", "log normal", "negated log normal"):
                start = {"mean": val,
                         "var": numpy.sqrt(numpy.abs(val) / 10.)}
            elif family in ("exponential", "negated exponential"):
                start = {"invrate": val}
            elif family in ("poisson", "negated poisson"):
                start = {"rate": val}
            else:
                raise Exception("Invalid parameter family: %s" % family)

            if self._verbose:
                start_str = ", ".join(["%s: %.3f" % (key, start[key])
                                       for key in start])
                print("\t %s %s {%s}" % (name, family, start_str))

            self._parameter[name + "_hyper"] = \
                HyperParameter(name, family, start)

            prior = self._get_parameter_prior(name)
            self._parameter[name] = [Parameter(name, j, family, prior)
                                     for j in range(n_groups)]

        ll = self._compute_parameter_loglikelihood()
        for name in self._parameter_name:
            for i in range(self._n_groups):
                self._parameter[name][i].log_likelihood = ll[i]

    def _find_starting_point(self):
        if self._verbose:
            print("Looking for a reasonable starting point: %s." %
                  datetime.datetime.now().strftime(DATETIMEFMT))

        ll = numpy.inf
        x = [0] * len(self._parameter_name)

        while not numpy.isfinite(ll):
            for i, name in enumerate(self._parameter_name):
                u = numpy.random.uniform(
                    low=self._parameter_value_range[name][0],
                    high=self._parameter_value_range[name][1])

                if self._parameter_family[name] == "poisson":
                    x[i] = int(round(u))
                else:
                    x[i] = u

            ll = self._mle_objective_function(x)

        self._starting_point = x

    def _mle_objective_function(self, x):
        valid = True
        for i, name in enumerate(self._parameter_name):
            if self._parameter_family[name] == "poisson":
                x[i] = int(round(x[i]))

            if (
                (x[i] < self._parameter_value_range[name][0]) or
                (self._parameter_value_range[name][1] < x[i])
            ):
                valid = False

        if not valid:
            return -1 * NEGATIVE_INFINITY

        param = [[p for i in range(self._n_groups)] for p in x]
        ll = self._loglikelihood_function(param)
        return -1 * numpy.sum(ll)

    def _optimise_starting_point(self):
        if self._verbose:
            print("Optimising a starting point: %s." %
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

            if res.fun < -0.9 * NEGATIVE_INFINITY:
                self._starting_point = res.x

                if res.success:
                    optimised = True

            else:
                self._find_starting_point()

            if n > 10:
                raise Exception("Cannot optimise a starting point. "
                                "Revise the value ranges")

    def _get_parameter_prior(self, name):
        return self._parameter[name + "_hyper"].get_distribution()

    def step(self, tune=False):
        for name in self._parameter_name:

            for i in range(self._n_groups):
                self._parameter[name][i].propose()

            ll = self._compute_parameter_loglikelihood(proposed_parameter=name)

            for i in range(self._n_groups):
                accepted = self._parameter[name][i].step(ll[i])

                if accepted:
                    for name_ in self._parameter_name:
                        self._parameter[name_][i].log_likelihood = ll[i]

                if tune:
                    self._parameter[name][i].tune()

            values = [self._parameter[name][i].value
                      for i in range(self._n_groups)]
            self._parameter[name + "_hyper"].update(values)

            prior = self._get_parameter_prior(name)
            for i in range(self._n_groups):
                self._parameter[name][i].set_prior(prior)

    def _compute_parameter_loglikelihood(self, proposed_parameter=None):

        x = [[0] * self._n_groups] * len(self._parameter_name)
        for i, name in enumerate(self._parameter_name):
            if name == proposed_parameter:
                x[i] = [p.proposal for p in self._parameter[name]]
            else:
                x[i] = [p.value for p in self._parameter[name]]

        # set the timeout handler
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self._loglikelihood_timeout)

        try:
            ll = self._loglikelihood_function(x)
        except TimeoutError:
            print("Likelihood computation timed out.")
            ll = numpy.zeros(sum(self._n_responses_per_group))
            ll += NEGATIVE_INFINITY
        finally:
            signal.alarm(0)

        return [sum(ll[self._ll_index[i]:self._ll_index[i + 1]])
                for i in range(self._n_groups)]

    @property
    def header(self):
        header = []
        for name in self._parameter_name:
            header.extend(self._parameter[name + "_hyper"].header)
            for parameter in self._parameter[name]:
                header.append(parameter.header)
        return header

    @property
    def values(self):
        values = []
        for name in self._parameter_name:
            values.extend(self._parameter[name + "_hyper"].value)
            for parameter in self._parameter[name]:
                values.append(parameter.value)
        return values

    @property
    def log_likelihoods(self):
        parameter_values = [[p.value for p in self._parameter[name]]
                            for name in self._parameter_name]
        ll = self._loglikelihood_function(parameter_values)

        return ll


class MCMC(object):
    def __init__(self, step_method, chain=0, sampledir=None,
                 display_progress=True, verbose=False):
        """
        MCMC class to sample from posterior.

        When sampledir is supplied, posterior samples and their log likelihoods
        are saved onto files. The log likelihood files are for computing the
        widely applicable information criteria and the leave-one-out
        cross-validation estimate, using loo package in R.

        :Arguments:

            - step_method : HierarchicalBayes

            - chain (optional) : int, default = 0
                This integer is used as part of file name to save samples and
                log likelihoods.

            - sampledir (optional) : str, default = None
                Where to save posterior samples. When it is None, samples are
                not saved into a file.

            - display_progress (optional) : bool, default = True
                Whether to display progress and estimated time of termination.

            - verbose (optional) : boolean, default = False

        """

        self._step_method = step_method
        self._chain = chain
        self._display_progress = display_progress
        self._verbose = verbose or False

        if sampledir is not None:
            sample_file = sampledir + "/sample.%i.csv" % chain
            ll_file = sampledir + "/log_likelihood.%i.csv" % chain

            i = 0
            while os.path.isfile(sample_file):
                sample_file = sampledir + "/sample.%i.%i.csv" % (chain, i)
                ll_file = sampledir + "/log_likelihood.%i.%i.csv" % (chain, i)
                i += 1

            # create empty files
            open(sample_file, "w").close()
            open(ll_file, "w").close()

        else:
            sample_file = None
            ll_file = None

        self._outputfile = {"sample": sample_file, "loglikelihood": ll_file}

    def sample(self, iter, burn=0, thin=1, tune_interval=100):
        """
        sample(iter, burn, thin, tune_interval, verbose)

        :Arguments:

          - iter : int
                Total number of iterations to do

          - burn (optional) : int, default = 0
                Variables will not be tallied until this many iterations are
                complete.

          - thin (optional) : int, default = 1
                Variables will be tallied at intervals of this many iterations.

          - tune_interval (optional) : int, default = 100
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
        self._tune_interval = int(tune_interval)

        if self._verbose:
            print("Chain %i sampling: %s." %
                  (self._chain,
                   datetime.datetime.now().strftime(DATETIMEFMT)))

        # Run sampler
        self._loop()

    def _loop(self):
        """
        Draws samples from the posterior.
        """

        self._start_time = datetime.datetime.now()
        display_interval = int(numpy.round(self._iter / 10.))

        tune = False
        # Loop
        for i in range(self._iter):

            # Tune at interval
            if i and (i < self._burn) and (i % self._tune_interval == 0):
                tune = True
            else:
                tune = False

            # take a step
            self._step_method.step(tune)

            if i == self._burn:
                self._print_header()

            # Record sample to trace, if appropriate
            if i % self._thin == 0 and i >= self._burn:
                self._print_sample(i)
                self._print_loglikelihood()

            if self._display_progress and (i % display_interval == 0):
                self._print_progress(i)

        if self._display_progress:
            self._print_progress(self._iter)
            print("")

    def _print_header(self):
        header = "index,chain," + ",".join(self._step_method.header)
        self._print("sample", header)

    def _print_sample(self, i):
        out = "%i,%i," % (i, self._chain)
        out += ",".join(["%f" % v for v in self._step_method.values])
        self._print("sample", out)

    def _print_loglikelihood(self):
        out = ",".join(["%f" % v for v in self._step_method.log_likelihoods])
        self._print("loglikelihood", out)

    def _print_progress(self, i):
        now = datetime.datetime.now()

        if i == 0:
            self._print(None, "\t  0%% complete at %s." %
                        now.strftime(DATETIMEFMT))
        elif i >= self._iter:
            elapsed = now - self._start_time
            self._print(None, "\t100%% complete at %s. Elapsed Time: %s" %
                        (now.strftime(DATETIMEFMT),
                         datetime.timedelta(seconds=elapsed.seconds)))
        else:
            percentage = float(i) / self._iter
            remain = (1 - percentage) * (now - self._start_time) / percentage
            self._print(None,
                        "\t%3i%% complete at %s. ETA: %s." %
                        (percentage * 100, now.strftime(DATETIMEFMT),
                         str((now + remain).strftime(DATETIMEFMT))))

    def _print(self, destination, output):
        if destination is None:
            print(output)
        elif self._outputfile[destination] is None:
            print(output)
        else:
            with open(self._outputfile[destination], "a") as h:
                h.write(output)
                h.write("\n")


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

        out = ",".join([key for key in self._summary.dtype.names])
        out += "\n"
        for row in self._summary:
            if hyperonly and (b"_" not in row[0]):
                continue

            out += "'%s',%.3f,%s,%.3f,%s,%.3f,%.3f,%.3f\n" % \
                   (row[0].decode("ascii"), row[1], row[2], row[3],
                    row[4], row[5], row[6], row[7])

        if csvfile is None:
            print(out.replace(",", ",\t"))

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
            self._hdi[key] = self._compute_hpd_interval(samples)

    def _compute_hpd_interval(self, samples):
        prob = self._hdi_p / 100.
        sorted_samples = numpy.array(sorted(samples))
        n_samples = len(samples)
        gap = max(1, min(n_samples - 1, round(n_samples * prob)))
        init = numpy.array(range(n_samples - gap))
        tmp = sorted_samples[init + gap] - sorted_samples[init]
        inds = numpy.where(tmp == min(tmp))[0][0]
        interval = (sorted_samples[inds], sorted_samples[inds + gap])

        return interval


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


def run_mcmc(parameter_name, parameter_family, parameter_value_range,
             n_groups, n_responses_per_group,
             loglikelihood_function,
             n_chains, n_iter, n_samples, outputdir,
             loglikelihood_timeout=60,
             start_with_mle=True,
             n_processes=0):
    """
    This function uses the classes defined above and runs MCMC.

    :Arguments:

        - parameter_name : tuple of str

            e.g., ("alpha", "beta")

        - parameter_family : dict

            e.g., {"alpha": "gaussian", "beta": "exponential"}

        - parameter_value_range : dict

            The parameter range to initialize a starting point. This is
            ignored during MCMC sampling.

            e.g., {"alpha": [0, 1], "beta": [2, 5]}

        - n_groups : int

        - n_responses_per_group : int or list of int

        - loglikelihood_function : def
            see doc string for HierarchicalBayes

        - n_iter: int
            Number of iteration per chain

        - n_samples: int
            Number of retained samples per chain

        - outputdir : str
            Full path specifying where to save MCMC samples, their summary, and
            figures (traceplots and bivariate). Before running MCMC, everything
            under this path will be removed.

        - loglikelihood_timeout (optional) : int, default = 60
            How many seconds to wait for likelihood computation. By default, if
            the loglikelihood_function does not return a value in 60 seconds,
            the likelihood is treated as 0. This may help speeding up MCMC
            sampling.

        - start_with_mle (optional) : bool, default = True
            Whether to start a chain with maximum likelihood estimation.
            This estimation pools groups and uses Nelder-Mead method.

        - n_processes (optional) : int, default = 0
            How many processes to launch. If zero (default), multiprocessing is
            not triggered.

    """

    assert(n_iter >= n_samples)
    print("Started at %s." %
          datetime.datetime.now().strftime(DATETIMEFMT))

    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)

    sampledir = outputdir + "/sample/"
    traceplotdir = outputdir + "/figure/traceplot/"
    bivariatedir = outputdir + "/figure/bivariate/"

    for directory in (sampledir, traceplotdir, bivariatedir):
        os.makedirs(directory, exist_ok=True)

    if n_iter // 2 > n_samples:
        burn = n_iter // 2
    else:
        burn = n_iter - n_samples

    thin = int(numpy.ceil((n_iter - burn) / n_samples))
    tune_interval = 100

    if n_processes <= 0:
        for chain in range(n_chains):
            _sample(
                parameter_name, parameter_family, parameter_value_range,
                n_groups, n_responses_per_group, loglikelihood_function,
                chain, n_iter, n_samples, burn, thin, tune_interval,
                sampledir, loglikelihood_timeout, start_with_mle)

    elif n_processes > 0:
        _parallel_sample(
            parameter_name, parameter_family, parameter_value_range,
            n_groups, n_responses_per_group, loglikelihood_function,
            n_chains, n_iter, n_samples, burn, thin, tune_interval,
            sampledir, loglikelihood_timeout, start_with_mle, n_processes)

    diagnostic = Diagnostic(sampledir)
    diagnostic.print(sampledir + "/summary.csv")
    diagnostic.print(hyperonly=True)

    fig = Figure(sampledir)
    fig.traceplots(traceplotdir)
    fig.bivariates(bivariatedir)


def _sample(parameter_name, parameter_family, parameter_value_range,
            n_groups, n_responses_per_group, loglikelihood_function,
            chain, n_iter, n_samples, burn, thin, tune_interval,
            sampledir, loglikelihood_timeout, start_with_mle,
            verbose=True):

    numpy.random.seed(chain)

    method = HierarchicalBayes(parameter_name,
                               parameter_family,
                               parameter_value_range,
                               n_groups,
                               n_responses_per_group,
                               loglikelihood_function,
                               loglikelihood_timeout,
                               start_with_mle=start_with_mle,
                               verbose=verbose)

    mcmc = MCMC(method, chain,
                sampledir=sampledir,
                display_progress=verbose,
                verbose=verbose)

    mcmc.sample(n_iter, burn, thin, tune_interval)


def _parallel_sample(parameter_name, parameter_family, parameter_value_range,
                     n_groups, n_responses_per_group, loglikelihood_function,
                     n_chains, n_iter, n_samples, burn, thin, tune_interval,
                     sampledir, loglikelihood_timeout, start_with_mle,
                     n_processes):

    processes = [Process(target=_sample,
                         args=(parameter_name, parameter_family,
                               parameter_value_range,
                               n_groups, n_responses_per_group,
                               loglikelihood_function,
                               chain, n_iter, n_samples,
                               burn, thin, tune_interval,
                               sampledir, loglikelihood_timeout,
                               start_with_mle,
                               (chain % n_processes) == 0))
                 for chain in range(n_chains)]

    processings = []
    for i, process in enumerate(processes):
        process.start()
        processings.append(process)

        if len(processings) % n_processes == 0:
            for processing in processings:
                processing.join()
            processings.clear()
