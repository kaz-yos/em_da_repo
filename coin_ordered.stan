/* Stan Code */
data {
    real<lower=0> a[2];
    real<lower=0> b[2];
    int<lower=0> N;
    int<lower=0> X[N];
}

parameters {
    /* ordered cannot be directly used with <lower=0,upper=1> */
    /* https://groups.google.com/forum/#!topic/stan-users/7r02EU7mL3o */
    ordered[2] log_odds_theta;
}

transformed parameters {
    vector[3]<lower=0,upper=1> theta;
    theta = inv_logit(log_odds_theta);
}

model {
    /* Prior's contribution to posterior log probability. */
    for (i in 1:2) {
        target += beta_lpdf(theta[i] | a[i], b[i]);
    }
    /* Data (likelihood)'s contribution to posterior log probability. */
    for (i in 1:N) {
        /* This part sums out the latent coin identity. */
        target += log_sum_exp(binomial_lpmf(X[i] | 10, theta[1]),
                              binomial_lpmf(X[i] | 10, theta[2]));
    }
}
