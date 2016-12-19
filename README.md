MCMC for Nested Data
====================

This module is written so that MCMC can be applied to nested data, using a
custom objective function. The module allows complete-, partial- and no-pooling
of nested data. The partial-pooling method is often called hierarchical Bayes
or multilevel modelling.

There exist several libraries to run MCMC (e.g., PyMC and Stan), but I could
not find one which suits my needs. While hierarchical Bayes is doable with the
existing libraries, those libraries often do not work with a custom likelihood
function. Even when they do, their sampling procedures are not optimized and
require a large number of calls to the objective function or the large number
of iterations for MCMC (as of mid-2015).  This is my attempt for a general
purpose MCMC module for nested data.

Parts of this module is influenced by PyMC, but I did not fork PyMC, because 1)
the procedure is quite a bit different and I could not think of a way to
integrate and 2) I wanted to cut off dependencies to the libraries with which
I am not familiar.


How to run examples
-------------------

The root directory is here (where this README.md is located).

```
python -m example.distribution
```


```
python -m example.regression
```

This second, regression example illustrates advantage of partial pooling.
Complete and no pooling result in variable estimate.


How to use this module
----------------------

To sample posterior, you need to import samplePosterior function in your
script.
```
from posteriorSampling import samplePosterior
```
Then call this function. Please see doc string for samplePosterior for more
details on how to call this function.

To diagnose samples, use diagnoseSamples imported from sampleDiagnosis.
```
from sampleDiagnosis import diagnoseSamples
```
Again, please see doc string for sampleDiagnosis for more details.



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
parameter values can be transformed within a objective function. If positive
values are desired, for example, you can exponentiate values. If integer is
desired, you can round.


LICENCE
-------

Copyright (C) 2015 2016 Takao Noguchi (tkngch@runbox.com)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.
