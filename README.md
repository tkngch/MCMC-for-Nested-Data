MCMC for Nested Data
====================

This module is written so that MCMC can be applied to nested data, using a
custom objective function. The module allows complete-, partial- and no-pooling
of nested data. The partial-pooling method is often called hierarchical Bayes.

There exist several libraries to run MCMC (e.g., PyMC and Stan), but I could
not find one which suits my needs. While hierarchical Bayes is doable with the
existing libraries, but those libraries often do not work with a custom
likelihood function. Even when they do, their sampling procedures are not
optimized and require a large number of calls to the objective function or the
large number of iterations for MCMC.  This is my attempt for a general purpose
MCMC module for nested data.

Parts of this module is influenced by PyMC, but I did not fork PyMC, because 1)
the procedure is quite a bit different and I could not think of a way to
integrate and 2) I wanted to cut off dependencies to the libraries with which
I am not familiar.


How to run examples
-------------------

The root directory is here.

```
python -m example.distribution
```

```
python -m example.regression
```


Notes on technical issues
-------------------------

- Gaussian assumption in partial-pooling.

For partial-pooling, individual parameters are modelled with Gaussian
distribution.  It is possible to use other distributions than Gaussian, such as
log-normal or exponential, but to my experience, non-Gaussian distributions
tend to lead to poorer parameter estimates. For example with exponential
distribution, individual parameters are too strongly pulled toward zero. With
log-normal distribution, parameter values are positively skewed: a few groups
tend to have extremely large parameter values.  I find that Gaussian
distribution tends to give better estimates.


- Prior for hyper-parameters in partial-pooling.

The partial-pooling requires prior for hyper-parameters, namely, mean and
standard deviation of Gaussian distribution.  This prior is set to improper
uniform.  This is because prior tends to have only minor impacts on estimate,
especially with the reasonable number of groups. If you want to put constraint
to the parameter values (e.g., when parameter values are desired to be
positive), complete- or no-pooling should be considered. Alternatively,
parameter values should be transformed within a objective function. If positive
values are desired, for example, you can exponentiate values. If integer is
desired, you can round.


LICENCE
-------

Copyright (C) 2015 Takao Noguchi (tkngch@runbox.com)

Apache?
