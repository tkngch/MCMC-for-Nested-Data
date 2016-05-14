#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2015-2016 Takao Noguchi (tkngch@runbox.com)

"""

MCMC with Hierarchical Bayes

This file declares and defines the classes and the functions to sample a
posterior distribution using MCMC algorithm.

"""

import signal
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
                "normal", "log normal", "negated log normal",
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
        if self._verbose:
            print("The new prior has mean of %.3f and var of %.3f."
                  % (prior.mean(), prior.var()))
        self._prior = prior

        if self._value is None:
            self.sample_prior_and_set_value()

        self._update_logprior()
        self._update_logp()

    def sample_prior_and_set_value(self):
        self._value = self._sample_prior()

    def _sample_prior(self):
        if "log" in self._family:
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
            value = -1 * value
        if "log" in self._family:
            value = numpy.log(value)

        if "poisson" in self._family:
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

        diff = proposal_logp - self._logp

        if (
            (not numpy.isfinite(self._logp)) and
            numpy.isfinite(proposal_logp)
        ):
            self._accept(proposal_log_likelihood)
            return True

        elif not numpy.isfinite(proposal_log_likelihood):
            self._reject()
            return False

        elif not numpy.isfinite(diff):
            self._reject()
            return False

        elif numpy.log(numpy.random.random()) < diff:
            self._accept(proposal_log_likelihood)
            return True

        self._reject()
        return False

    def _accept(self, proposal_log_likelihood):
        self._value = self._proposal
        self._log_likelihood = proposal_log_likelihood

        self._update_logprior()
        self._update_logp()

        self._n_accepted += 1.
        if self._verbose > 0:
            print("accepted")
        self._proposal = None

    def _reject(self):
        self._n_rejected += 1.
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
                 start=None, prior=None, verbose=False):
        """

        This class defines hyper-parameter. As with Parameter class, users are
        not expected to directly interact with this class.

        :Arguments:
            - parameter_name : str

            - family : str
                "normal", "log normal", "negated log normal",
                "exponential", "negated exponential",
                "poisson", or "negated poisson".

                This family defines how parameter values are modeled. With the
                normal family, for example, parameter values are assumed to
                be normally distributed, and its mean and variance become
                hyper-parameters.

                Naturally, this family also defines range of
                parameter values: with normal family, a parameter value can
                take any real values, but with log normal or exponential
                family, a parameter to value can only be a positive real. With
                poisson family, the parameter value can take only non-negative
                integers.

                In modeling non-positive parameter values, negated family may
                come useful. Negated exponential, for example, is -1 *
                exponential distribution, which takes non-positive values.

            - start (optional) : dict, default = None
                default values are:
                    {"mu": 0, "sigma2": 1} for normal family
                    {"mu": 0, "sigma2": 1} for log normal family
                    {"invrate": 1} for exponential family
                    {"rate": 1} for poisson family

            - prior (optional) : dict, default = None

                For normal and log normal families, prior parameters are
                "mu0", "kappa0", "nu0", and "sigma20". Then, prior is give by:

                    normal family:
                        mu | sigma2 ~ Normal(mu0, sigma2 / kappa0)
                        sigma ~ Uniform(0, inf)

                    log normal family:
                        mu | sigma2 ~ Normal(mu0, sigma2 / kappa0)
                        sigma ~ Uniform(0, inf)

                It is tempting to use the inverse Chisquare for sigma2:
                        sigma2 ~ Scaled-Inv-Chisq(nu0, sigma20).
                But this prior leads to improper posterior when nu0 and sigma20
                are both close to zero. See page 117 on BDA3.

                The interpretations of the above parameters are: mean is
                estimated from observations with total precision (sum of all
                individual precisions) kappa0 / sigma2 and with sample mean
                mu0.  The default is {"mu0": 0, "kappa0": 0}.  Thus, the
                default is improper non-informative.  For more details, see
                pages 67-69 in BDA3.

                For exponential faimily, prior parameters are "alpha0" and
                "beta0".
                    rate ~ Gamma(alpha0, beta0)
                This parameterisation is equivalent to a total count of alpha0
                observations with sum beta0. The default is {"alpha0": 0,
                "beta0": 0}. For more details, see page 46 in BDA3.

                For poisson family, prior parameters are "alpha" and "beta".
                    rate ~ Gamma(alpha0, beta0)
                This parameterisation is equivalent to a total count of alpha0
                in beta0 prior observations. The default is {"alpha0": 0,
                "beta0": 0}. For more details, see pages 43-44 in BDA3.

            - verbose (optional) : bool, default = False

        """

        self._parameter_name = parameter_name
        self._family = family

        if start is not None:
            self._value = start
        else:
            self._value = {
                "normal": {"mu": 0, "sigma2": 1},
                "log normal": {"mu": 0, "sigma2": 1},
                "negated log normal": {"mu": 0, "sigma2": 1},
                "exponential": {"invrate": 1},
                "negated exponential": {"invrate": -1},
                "poisson": {"rate": 1},
                "negated poisson": {"rate": -1}
            }[self._family]

        if prior is not None:
            self._prior = prior
        elif self._family in ("normal", "log normal", "negated log normal"):
            self._prior = {"mu0": 0, "kappa0": 0, "nu0": 0, "sigma20": 0}
        elif self._family in ("exponential", "negated exponential",
                              "poisson", "negated poisson"):
            self._prior = {"alpha0": 0, "beta0": 0}
        else:
            raise Exception("Invalid family: ", self._family)

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
        if "log" in self._family:
            x = numpy.log(x)

        if "normal" in self._family:
            self._update_var(x)
            self._update_mean(x)

        elif "exponential" in self._family:
            self._update_rate_param(len(x), sum(x))
            self._value["invrate"] = 1. / self._value["rate"]

        elif "poisson" in self._family:
            self._update_rate_param(sum(x), len(x))

        if "negated" in self._family:
            for key in ("mu", "invrate", "rate"):
                if key in self._value:
                    self._value[key] *= -1

        if self._verbose:
            print("\tupdated: %s" % self.values.__str__())

    def _update_mean(self, x):
        """
        See pages 68, 116-117, and 289 on BDA3.
        """
        n = len(x)

        mu_n = (self._prior["kappa0"] * self._prior["mu0"] +
                n * numpy.mean(x)) / (self._prior["kappa0"] + n)
        kappa_n = self._prior["kappa0"] + n

        sd = numpy.sqrt(self._value["sigma2"] / kappa_n)
        self._value["mu"] = numpy.random.normal(mu_n, sd)

    def _update_var(self, x):
        """
        See pages 189-190 on BDA3.
        """
        n = len(x)
        hat = numpy.sum((x - self._value["mu"]) ** 2) / (n - 1)
        self._value["sigma2"] = self._sample_inv_chi2(n - 1, hat)

    def _sample_inv_chi2(self, v, s2):
        return scipy.stats.invgamma(v / 2., scale=(v / 2.) * s2).rvs()

    def _update_rate_param(self, shape, inv_scale):
        shape += self._prior["alpha0"]
        inv_scale += self._prior["beta0"]
        self._value["rate"] = \
            scipy.stats.gamma(a=shape, scale=1./inv_scale).rvs()

    def get_distribution(self):
        if self._family == "normal":
            return scipy.stats.norm(loc=self._value["mu"],
                                    scale=numpy.sqrt(self._value["sigma2"]))

        if self._family == "log normal":
            return scipy.stats.norm(loc=self._value["mu"],
                                    scale=numpy.sqrt(self._value["sigma2"]))

        if self._family == "negated log normal":
            return scipy.stats.norm(loc=-1 * self._value["mu"],
                                    scale=numpy.sqrt(self._value["sigma2"]))

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
                 prior_for_hyperparameter=None, loglikelihood_timeout=60,
                 start_with_mle=False, verbose=False):
        """
        This class defines the step method and finds a starting state for MCMC.

        :Arguments:

            - parameter_name : tuple of str
                e.g., ("alpha", "beta")

            - parameter_family : dict
                e.g., {"alpha": "normal", "beta": "exponential"}

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

            - prior_for_hyperparameter (optional) : dict, default = None
                See doc strings for sample_posterior and HyperParameter.

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

        if prior_for_hyperparameter is None:
            prior_for_hyperparameter = \
                dict((name, None) for name in self._parameter_name)

        self._n_groups = n_groups

        if type(n_responses_per_group) == int:
            self._n_responses_per_group = [n_responses_per_group] * n_groups
        elif type(n_responses_per_group) == list:
            self._n_responses_per_group = n_responses_per_group

        self._loglikelihood_function = loglikelihood_function
        self._loglikelihood_timeout = loglikelihood_timeout

        self._ll_index = numpy.hstack(
            [0, numpy.cumsum(self._n_responses_per_group)])
        assert(len(self._ll_index) == self._n_groups + 1)

        self._find_starting_point()
        if start_with_mle:
            self._optimise_starting_point()

        if self._verbose:
            print("Starting state:")

        self._initialise_parameters(prior_for_hyperparameter)
        self._determine_starting_point()

        if self._verbose:
            print("")

    def _find_starting_point(self):
        if self._verbose:
            print("Started looking for a reasonable starting state at %s." %
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

        if self._verbose:
            print("Found a starting state at %s." %
                  datetime.datetime.now().strftime(DATETIMEFMT),
                  end="\n\n")

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
            return -1 * float("inf")

        param = [[p for i in range(self._n_groups)] for p in x]
        ll = self._loglikelihood_function(param)
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

    def _initialise_parameters(self, prior_for_hyperparameter):
        self._parameter = {}
        for i, name in enumerate(self._parameter_name):

            family = self._parameter_family[name]
            val = self._starting_point[i]

            # if family in ("normal", "log normal", "negated log normal"):
            if family in ("normal",):
                start = {"mu": val,
                         "sigma2": numpy.sqrt(numpy.abs(val) / 10.)}
            elif family in ("log normal", "negated log normal"):
                start = {"mu": numpy.sign(val) * numpy.log(abs(val)),
                         "sigma2": numpy.sqrt(numpy.abs(val) / 10.)}
            elif family in ("exponential", "negated exponential"):
                start = {"invrate": val}
            elif family in ("poisson", "negated poisson"):
                start = {"rate": val}
            else:
                raise Exception("Invalid parameter family: %s" % family)

            if self._verbose:
                start_str = ", ".join(["%s: %.8f" % (key, start[key])
                                       for key in start])
                print("\t %s %s {%s}" % (name, family, start_str))

            if name not in prior_for_hyperparameter:
                prior = None
            else:
                prior = prior_for_hyperparameter[name]

            self._parameter[name + "_hyper"] = \
                HyperParameter(name, family, start=start, prior=prior)

            prior = self._get_parameter_prior(name)
            self._parameter[name] = [Parameter(name, j, family, prior)
                                     for j in range(self._n_groups)]

    def _determine_starting_point(self):
        if self._verbose:
            print("Determining individual startin states at %s." %
                  datetime.datetime.now().strftime(DATETIMEFMT))

        ll = [-1 * float("inf")] * self._n_groups

        while not all(numpy.isfinite(ll)):
            ll = self._compute_parameter_loglikelihood()

            for name in self._parameter_name:
                for i in range(self._n_groups):
                    if numpy.isfinite(ll[i]):
                        self._parameter[name][i].log_likelihood = ll[i]
                    else:
                        self._parameter[name][i].sample_prior_and_set_value()

    def _get_parameter_prior(self, name):
        return self._parameter[name + "_hyper"].get_distribution()

    def step(self, tune=False):
        for name in self._parameter_name:

            for i in range(self._n_groups):
                self._parameter[name][i].propose()

            llarray = self._compute_parameter_loglikelihood(
                proposed_parameter=name)

            for i, ll in enumerate(llarray):
                accepted = self._parameter[name][i].step(ll)

                if accepted:
                    for name_ in self._parameter_name:
                        self._parameter[name_][i].log_likelihood = ll

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
            ll += float("inf")
        finally:
            signal.alarm(0)

        # j = 15
        # print("parameter: [%s], ll: %s"
        #       % (", ".join(["%f" % x[k][j] for k in range(len(self._parameter_name))]),
        #          ", ".join(["%f" % l for l in ll[self._ll_index[j]:self._ll_index[j + 1]]])))

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
    def __init__(self, step_method, chain, sampledir=None,
                 save_loglikelihoods=True,
                 display_progress=True, verbose=False):
        """
        MCMC class to sample from posterior.

        When sampledir is supplied, posterior samples and their log likelihoods
        are saved onto files. The log likelihood files are for computing the
        widely applicable information criteria and the leave-one-out
        cross-validation estimate, using loo package in R.

        :Arguments:

            - step_method : HierarchicalBayes

            - sampledir (optional) : str, default = None
                Where to save posterior samples. When it is None, samples are
                not saved into a file.

            - save_loglikelihoods (optional) : bool, defalt = True
                Whether to save loglikelihood when saving samples. Saved
                loglikelihood can be used to compute waic or loo with loo
                package in R.

            - display_progress (optional) : bool, default = True
                Whether to display progress and estimated time of termination.

            - verbose (optional) : boolean, default = False

        """

        self._step_method = step_method
        self._chain = chain
        self._save_loglikelihoods = save_loglikelihoods
        self._display_progress = display_progress
        self._verbose = verbose or False

        if sampledir is not None:
            i = 0
            sample_file = sampledir + "/sample.%i.csv" % i
            ll_file = sampledir + "/log_likelihood.%i.csv" % i

            while os.path.isfile(sample_file):
                i += 1
                sample_file = sampledir + "/sample.%i.csv" % i
                ll_file = sampledir + "/log_likelihood.%i.csv" % i

            # create empty files
            open(sample_file, "w").close()

            if self._save_loglikelihoods:
                open(ll_file, "w").close()
            else:
                ll_file = None

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
            print("Chain %i. Sampling initialised at %s." %
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

                if self._save_loglikelihoods:
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
                         datetime.timedelta(
                            seconds=int(elapsed.total_seconds()))))
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


def sample_posterior(parameter_name, parameter_family, parameter_value_range,
                     n_groups, n_responses_per_group, loglikelihood_function,
                     n_chains, n_iter, n_samples, outputdir,
                     prior_for_hyperparameter=None, start_with_mle=True,
                     chain_seed=None, loglikelihood_timeout=60,
                     n_processes=None, clear_outputdir=True,
                     save_loglikelihoods=True, verbose=True):
    """
    This function uses the classes defined above and runs MCMC.

    :Arguments:

        - parameter_name : tuple of str

            e.g., ("alpha", "beta")

        - parameter_family : dict

            e.g., {"alpha": "normal", "beta": "exponential"}

        - parameter_value_range : dict

            The parameter range to initialize a starting point. This is
            ignored during MCMC sampling.

            e.g., {"alpha": [0, 1], "beta": [2, 5]}

        - n_groups : int

        - n_responses_per_group : int or list of int

        - loglikelihood_function : def
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

        - start_with_mle (optional) : bool, default = True
            Whether to start a chain with maximum likelihood estimation.
            This estimation pools groups and uses Nelder-Mead method.

        - chain_seed (optional) : iterable, default = range(n_chains)
            Seed for random number generation. The length of the seeds has to
            be the same as the number of chains.

        - loglikelihood_timeout (optional) : int, default = 60
            How many seconds to wait for likelihood computation. By default, if
            the loglikelihood_function does not return a value within 60
            seconds, the likelihood is treated as 0. This may help speeding up
            MCMC sampling.

        - n_processes (optional) : int, default = 1
            How many processes to launch. If 1 (default), multiprocessing is
            not triggered. If zero, all the available cores are used.

        - clear_outputdir (optional) : bool, default = True
            Whether to delete all the files in the outputdir.

        - save_loglikelihoods (optional) : bool, defalt = True
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

    sampledir = outputdir + "/sample/"
    os.makedirs(sampledir, exist_ok=True)

    if n_iter // 2 > n_samples:
        burn = n_iter // 2
    else:
        burn = n_iter - n_samples

    thin = int(numpy.ceil((n_iter - burn) / n_samples))
    tune_interval = 100

    if n_processes == 0:
        n_processes = min(n_chains, multiprocessing.cpu_count())

    if n_processes == 1:
        for chain, seed in enumerate(chain_seed):
            _sample(
                parameter_name, parameter_family, parameter_value_range,
                n_groups, n_responses_per_group, loglikelihood_function,
                prior_for_hyperparameter,
                seed, chain, n_iter, n_samples, burn, thin, tune_interval,
                sampledir, loglikelihood_timeout, start_with_mle,
                save_loglikelihoods, verbose)

    elif n_processes > 1:
        _parallel_sample(
            parameter_name, parameter_family, parameter_value_range,
            n_groups, n_responses_per_group, loglikelihood_function,
            prior_for_hyperparameter,
            chain_seed, n_iter, n_samples, burn, thin, tune_interval,
            sampledir, loglikelihood_timeout, start_with_mle, n_processes,
            save_loglikelihoods, verbose)

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


def _sample(parameter_name, parameter_family, parameter_value_range,
            n_groups, n_responses_per_group, loglikelihood_function,
            prior_for_hyperparameter,
            seed, chain, n_iter, n_samples, burn, thin, tune_interval,
            sampledir, loglikelihood_timeout, start_with_mle,
            save_loglikelihoods, verbose):

    numpy.random.seed(seed)

    method = HierarchicalBayes(
        parameter_name, parameter_family, parameter_value_range,
        n_groups, n_responses_per_group, loglikelihood_function,
        prior_for_hyperparameter=prior_for_hyperparameter,
        loglikelihood_timeout=loglikelihood_timeout,
        start_with_mle=start_with_mle,
        verbose=verbose)

    mcmc = MCMC(
        method, chain,
        sampledir=sampledir,
        save_loglikelihoods=save_loglikelihoods,
        display_progress=verbose,
        verbose=verbose)

    mcmc.sample(n_iter, burn, thin, tune_interval)


def _parallel_sample(parameter_name, parameter_family, parameter_value_range,
                     n_groups, n_responses_per_group, loglikelihood_function,
                     prior_for_hyperparameter,
                     chain_seed, n_iter, n_samples, burn, thin, tune_interval,
                     sampledir, loglikelihood_timeout, start_with_mle,
                     n_processes, save_loglikelihoods, verbose):

    n_chains = len(chain_seed)

    processes = [Process(target=_sample,
                         args=(parameter_name, parameter_family,
                               parameter_value_range,
                               n_groups, n_responses_per_group,
                               loglikelihood_function,
                               prior_for_hyperparameter,
                               chain_seed[chain], chain, n_iter, n_samples,
                               burn, thin, tune_interval,
                               sampledir, loglikelihood_timeout,
                               start_with_mle,
                               save_loglikelihoods,
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
