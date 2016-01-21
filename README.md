MCMC for Hierarchical Bayes in Python
=====================================

This module is written so that hierarchical Bayes can be applied with a custom
objective function. There exist several libraries to run MCMC (e.g., PyMC and
Stan), but I could not find one which suits my needs. While hierarchical Bayes
is doable with the existing libraries, but those libraries often do not work
with a custom likelihood function. Even when they do, their sampling procedures
are not optimized and require a large number of iterations for MCMC.  This is
my attempt for a general purpose MCMC module for hierarchical Bayes.

Parts of this module is influenced by PyMC, but I did not fork PyMC, because 1)
the procedure is quite a bit different and I could not think of a way to
integrate and 2) I wanted to cut off dependencies to the libraries with which
I am not familiar.


LICENCE
-------

Copyright (C) 2015 Takao Noguchi (tkngch@runbox.com)

Apache?
