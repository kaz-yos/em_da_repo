/* Stan Code */
data {
    real<lower=0> a[2];
    real<lower=0> b[2];
    int<lower=0> N;
    int<lower=0> X[N];
}

parameters {
    real<lower=0,upper=1> theta[2];
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
