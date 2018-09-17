# Dealing with censored data using STAN

### Censoring VS truncating

* **Censoring** happens when values outside a pre-defined window are replaced by the closest values of the boundary of the window.

* **Truncation** happens when said points are discarded from the data set.


### STAN

STAN is a high-level language that allows defining generative probabilistic models, which, in turn, are sampled with a Hamiltonian Monte Carlo method performed by STAN's engine.

Defining **truncated** version of 1-D distributions is directly implemented in STAN with the `T[...]` notation, e.g. `x ~ normal(mu, sigma) T[low, high];` describes sampling `x` from a truncated 1-D normal distribution with mean `mu`, stdev `sigma`, truncated between `low` and `high` values.

```
model {
    x ~ normal(mu, sigma) T[low, high];
}
```

Defining **censoring**, even in 1-D is not so straigthforward: The degenerate distribution of censored points need to be added manually using conditions and the cumulative (`_lcdf`) and complement-cumulative (`_lccdf`) versions of the distribution:

```
model {
    if ( x <= low ) {
        target += normal_lcdf(low | mu, sigma);
    }
    else if ( x >= high ) {
        target += normal_lccdf(high | mu, sigma);
    }
    else {
        x ~ normal(mu, sigma);
    }
}
```

**Problem:** In more than 1 dimension, none of the above methods work. Stan does not support truncated version and of multi-variate distributions and no "2D cdf" (whatever that may be) is implemented.


### Contents of this repo
Two notebooks and 5 stan model code files exemplify the censoring and truncation problem for 1- and 2-D normal distributions, and provide different solutions.

Notebooks (in notebooks/ directory):

* **01-Generate-data.ipynb**: python notebook to generate original and censored 2-D normal data
* **02-Fit-models.ipynb**: Fitting 3 STAN models to the generated data

Stan models (in stan_code/ directory):

* **1D-truncated-normal.stan**: 1D truncated normal model
* **1D-censored-normal.stan**: 1D censored normal model
* **2D-truncated-normal.stan**: 2D truncated normal model
* **2D-censored-normal-marginalized.stan**: 2D censored normal model (with exact marginalization of conditional y|x)
* **2D-censored-normal-w-1D-latent-optimized.stan**: 2D censored normal model (with latent variables)


### Attribution
* **Author:** Peter Komar
* **GitHub repo:** https://github.com/peterkomar-hu/censoring-w-stan
* **License:** This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>. <br>
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a>