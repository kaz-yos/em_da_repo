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
        target += beta_lpdf(theta[j] | a[j], b[j]);
        /* Need for Jacobian adjustment? */
        /* https://mc-stan.org/docs/2_19/stan-users-guide/change-of-variables-chapter.html */
        /* https://discourse.mc-stan.org/t/putting-priors-on-transformed-parameters/2488 */
        /* https://groups.google.com/forum/#!topic/stan-users/sheMJeXXNL4 */
        /*  */
        /* Left-hand side of sampling statement (~) may contain a non-linear */
        /* transform of a parameter or local variable. If it does, you need */
        /* to include a target += statement with the log absolute */
        /* determinant of the Jacobian of the transform. */
        /* d/dtheta log(theta/(1-theta)) = 1/(theta(1-theta)) */
        /*  */
        /* NO NEED IN THIS CASE???? <= Not sure */
        /* We started with theta that we assumed Beta(a,b) on and created logit(theta). */
        /* Thus, we are specifying a correct distribution on theta. */
    }
    /* Data (likelihood)'s contribution to posterior log probability. */
    for (i in 1:N) {
        /* This part sums out the latent coin identity. */
        target += log_sum_exp(binomial_lpmf(X[i] | 10, theta[1]),
                              binomial_lpmf(X[i] | 10, theta[2]));
    }
}
