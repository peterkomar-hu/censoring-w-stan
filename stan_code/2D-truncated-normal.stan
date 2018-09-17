// This STAN code implements a 2D normal distribution truncated along the y axis
//
// Inputs:
//      y_low, y_high: censoring boundaries
//      x: 1D array of x values
//      y: 1D array of censored y values
//
// Parameters:
//      mu: mean of 2D normal distribution
//      Sigma: covariance matrix of 2D normal distribution


data {
    real eps;     // numerical margin
    real y_low;   // lower censoring level
    real y_high;  // upper censoring level

    int<lower=0> N;                                         // number of data points
    real x[N];                                              // x values
    real<lower = y_low - eps, upper = y_high + eps> y[N];   // y values
}

transformed data {
    real x_bottom[N];     // x of points on the lower censoring level
    vector[2] xy_mid[N];  // (x,y) of non-censored points
    real x_top[N];        // x of points on the upper censoring level

    int N_bottom = 0;
    int N_mid = 0;
    int N_top = 0;
    for (n in 1:N){
        if (y[n] <= y_low + eps){         // Appending to x_bottom array
            N_bottom += 1;
            x_bottom[N_bottom] = x[n];

        }
        else if (y[n] >= y_high - eps){   // Appending to x_top array
            N_top += 1;
            x_top[N_top] = x[n];
        }
        else {                            // Appending to xy_mid array
            N_mid += 1;
            xy_mid[N_mid][1] = x[n];
            xy_mid[N_mid][2] = y[n];
        }
    }
}

parameters {
    vector[2] mu;           // mean of 2D normal distribution
    cov_matrix[2] Sigma;    // covariance matrix of 2D normal distribution
}

model {
    real mu_y_given_x;     // mean of y | x, which is different for each x data point
    real sigma_y_given_x;  // stdev of y | x
    
    Sigma[2,2] ~ lognormal(0, 4.6);  // prior for Var(y)
    
    // For the conditional mu and sigma values we use the formulas in 
    // https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    
    sigma_y_given_x = sqrt(Sigma[2,2] - Sigma[2,1] * Sigma[1,2] / Sigma[1,1]);      // stdev of y | x  (constant over all x)
    for (n in 1:N_mid) {
        xy_mid[n][1] ~ normal(mu[1], sqrt(Sigma[1,1]));                             // 1D normal for x
        mu_y_given_x = mu[2] + (xy_mid[n][1] - mu[1]) * Sigma[2,1] / Sigma[1,1];    // mean of y | x
        xy_mid[n][2] ~ normal(mu_y_given_x, sigma_y_given_x) T[y_low, y_high];      // truncated 1D normal for y
    }
}
