// This STAN model implements a 1D truncated normal distribution
//
// Input data:
//      * 1D array of real numbers
//
// Parameters:
//      * mu: mean of the normal distribution
//      * sigma: stdev of the normal distribution


data {
    real low;                           // lower censoring bound
    real<lower=low> high;               // upper censoring bound
    real<lower=0> eps;                  // numerical margin around the edges determining  which points are considered being on the boundary
    int<lower=0> N;                     // size of main input
    real<lower=low, upper=high> x[N];   // main input
}

transformed data {
    int N_low = 0;      // number of points on the lower boundary
    int N_high = 0;     // number of points on the upper boundary
    int N_mid = 0;      // number of points inside the censoring window
    real x_mid[N];      // values of points inside the censoring window
    
    for (n in 1:N){
        if (x[n] <= low + eps){
            N_low += 1;
        }
        else if (x[n] >= high - eps){
            N_high += 1;
        }
        else {
            N_mid += 1;
            x_mid[N_mid] = x[n];
        }
    }
}

parameters {
    real mu;                // mean of normal distribution
    real<lower=0> sigma;    // stdev of normal distribution
}

model {

    mu ~ cauchy(0, 10);         // prior for mu
    sigma ~ lognormal(0, 2.3);  // prior for sigma

    for (n in 1:N_mid){
        x_mid[n] ~ normal(mu, sigma) T[low, high];  // truncated normal model
    }
}
