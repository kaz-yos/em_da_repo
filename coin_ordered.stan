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
    /* https://mc-stan.org/docs/2_19/stan-users-guide/reparameterizations.html */
    /* This means that HMC works with the density function for log_odds_theta? */
    ordered[2] log_odds_theta;
}

transformed parameters {
    real<lower=0,upper=1> theta[2];
    for (j in 1:2) {
        theta[j] = inv_logit(log_odds_theta[j]);
    }
}

model {
    /* Prior's contribution to posterior log probability. */
    for (j in 1:2) {
        /* Need for Jacobian adjustment */
        /*  */
        /* 20.3 Changes of Variables (See the second inv-gamma/gamma example) */
        /* https://mc-stan.org/docs/2_19/stan-users-guide/changes-of-variables.html */
        /*  */
        /* Putting priors on transformed parameters */
        /* https://discourse.mc-stan.org/t/putting-priors-on-transformed-parameters/2488 */
        /* Stan users mailing list Prior on transformed parameters? */
        /* https://groups.google.com/forum/#!topic/stan-users/sheMJeXXNL4 */
        /*  */
        /* Left-hand side of sampling statement (~) may contain a non-linear */
        /* transform of a parameter or local variable. If it does, you need */
        /* to include a target += statement with the log absolute */
        /* determinant of the Jacobian of the transform. */
        /*  */
        /* We have to make the distribution of log_odss_theta contribute. */
        /* Let eta = log(theta / (1 - theta)) */
        /* theta = e^eta / (1 + e^eta) = expit(eta) */
        /* Check: https://www.wolframalpha.com/input/?i=e%5Ex+%2F+(1+%2Be%5Ex) */
        /* d theta / d eta = d expit(eta) / d eta = e^eta / (1 + e^eta)^2 */
        /* f_eta(eta) = f_theta(expit(eta)) * |d expit(eta) / d eta| */
        /*            = f_theta(theta) * (e^eta / (1 + e^eta)^2) */
        /*  */
        /* log density contributions */
        /* log f_theta(theta) can be evaluated using beta_lpdf */
        /* log (e^eta / (1 + e^eta)^2) = eta - 2 * log(1 + e^eta) */
        /*  */
        /* Beta part */
        target += beta_lpdf(theta[j] | a[j], b[j]);
        /* Jacobian part */
        target += log_odds_theta[j] - 2 * log(1 + exp(log_odds_theta[j]));
    }
    /*  */
    /* Data (likelihood)'s contribution to posterior log probability. */
    for (i in 1:N) {
        /* This part sums out the latent coin identity. */
        target += log_sum_exp(binomial_lpmf(X[i] | 10, theta[1]),
                              binomial_lpmf(X[i] | 10, theta[2]));
    }
}
