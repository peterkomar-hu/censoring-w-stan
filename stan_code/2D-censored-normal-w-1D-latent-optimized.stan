// This STAN code implements a 2D normal distribution censored along the y axis
// It samples one latent, 1D variable for each point on the boundaries
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
    vector[2] mu;                           // mean of Gaussian
    cov_matrix[2] Sigma;                    // covariance of Gaussian
    real<upper=y_low> z_bottom[N_bottom];   // latent y values below y_low
    real<lower=y_high> z_top[N_top];        // latent y values above y_high
}

model {
    vector[2] xz_bottom[N_bottom];                  // (x,z) points below the lower boundary
    vector[2] xz_top[N_top];                        // (x,z) points above the upper boundary
    
    for (n in 1:N_bottom){
        xz_bottom[n][1] = x_bottom[n];
        xz_bottom[n][2] = z_bottom[n];
        xz_bottom[n] ~ multi_normal(mu, Sigma);     // 2D normal of (x,z) points
    }

    for (n in 1:N_top){
        xz_top[n][1] = x_top[n];
        xz_top[n][2] = z_top[n];
        xz_top[n] ~ multi_normal(mu, Sigma);        // 2D normal of (x,z) points
    }

    for (n in 1:N_mid) {
        xy_mid[n] ~ multi_normal(mu, Sigma);        // 2D normal of (x,y) points
    }
}



