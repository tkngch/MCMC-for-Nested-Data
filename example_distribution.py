#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import numpy
import scipy.stats
from posterior_sampling import sample_posterior
from diagnosis import diagnose
numpy.random.seed(12345)


def get_objective_func(parameter_name, parameter_family,
                       n_groups, n_responses_per_group):

    def _compute_loglikelihood(parameter,
                               n_groups, n_responses_per_group,
                               distributions):
        ll = numpy.zeros(n_groups * n_responses_per_group)
        i = 0
        for j in range(n_groups):
            for k in range(n_responses_per_group):
                ll[i] = sum([dist[j].logpdf(parameter[n][j])
                             for n, dist in enumerate(distributions)])
                i += 1

        return ll

    print("True value:")

    distributions = []
    for i, name in enumerate(parameter_name):
        scale = numpy.random.random() * 1
        if "negated" in parameter_family[name]:
            scale *= -1

        mu = [(i + 1) * scale + j / float(n_groups) for j in range(n_groups)]
        sd = numpy.sqrt(abs(scale))

        tmp = [scipy.stats.norm(loc=loc, scale=sd) for loc in mu]
        distributions.append(tmp)

        print("\t%s: {mean: %.2f, scale: %.2f}"
              % (name, numpy.mean(mu), scale))
    print("")

    func = functools.partial(_compute_loglikelihood,
                             n_groups=n_groups,
                             n_responses_per_group=n_responses_per_group,
                             distributions=distributions)

    return func


def main():
    n_chains = 2
    n_iter = 10000
    n_samples = 1000
    outputdir = "./sample/example_distribution/"

    # parameter_name = ("a", "b", "c", "d")
    parameter_name = ("a", "b", "e",)
    parameter_family = {"a": "normal",
                        "b": "log normal",
                        "c": "exponential",
                        "d": "poisson",
                        "e": "negated log normal",
                        "f": "negated exponential",
                        "g": "negated poisson"}

    parameter_value_range = {}
    for name in ("a", "b", "c", "d"):
        parameter_value_range[name] = [0, 100]
    for name in ("e", "f", "g"):
        parameter_value_range[name] = [-100, 0]

    n_groups = 10
    n_responses_per_group = 10

    objective_func = get_objective_func(parameter_name, parameter_family,
                                        n_groups, n_responses_per_group)

    sample_posterior(parameter_name, parameter_family, parameter_value_range,
                     n_groups, n_responses_per_group, objective_func,
                     n_chains, n_iter, n_samples, outputdir,
                     start_with_mle=False, n_processes=0)

    diagnose(outputdir)


if __name__ == "__main__":
    main()
