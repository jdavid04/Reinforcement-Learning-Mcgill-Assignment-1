import numpy as np


def action_elimination(bandit_means, bandit_scales, samples_per_epoch=1,
                       epsilon=0.01, delta=0.1):
    """
    Arguments:
        - bandit_means: List of Gaussian random variable means
        - bandit_scales : List of Gaussian random variable
                             standard deviations (same order)
        - samples_per_epoch : number of times to sample each arm/epoch
        - epsilon : parameter for LIL bound
        - delta : parameter for LIL bound and confidence level
    """

    n = len(bandit_means)
    bandit_samples = {key: [] for key in range(n)}
    bandit_estimates = [None]*n
    bandit_bounds = [None]*n
    converged = False
    bandit_set = set(range(n))

    num_epochs = 1
    while not converged:
        for arm in bandit_set:
            sample = np.random.normal(bandit_means[arm], bandit_scales[arm],
                                      size=samples_per_epoch)
            bandit_samples[arm].extend(sample)
            bandit_estimates[arm] = np.mean(bandit_samples[arm])

        bound = _compute_bound(t=samples_per_epoch*num_epochs,
                               eps=epsilon, delta=delta/n)
        max_ref = np.max(bandit_estimates)
        index_mask = set([])

        for arm in bandit_set:
            if bandit_estimates[arm]+bound <= max_ref - bound:
                index_mask.add(arm)
        bandit_set -= index_mask

        if len(bandit_set) == 1:
            converged = True
        num_epochs += 1

    return bandit_set


def ucb(bandit_means, bandit_scales, epsilon=0.01, delta=0.1, beta=1):
    """
    Arguments:
        - bandit_means: List of Gaussian random variable means
        - bandit_scales : List of Gaussian random variable
                             standard deviations (same order)
        - epsilon : parameter for LIL bound
        - delta : parameter for LIL bound and confidence level
        - beta : parameter for bound
    """

    n = len(bandit_means)
    m = ((2+beta)/beta)**2
    alpha = m * (1 + (np.log(2*np.log(m*n/delta))/np.log(n/delta)))
    bandit_samples = {key: [np.random.normal(bandit_means[key],
                      bandit_scales[key])] for key in range(n)}
    bandit_estimates = [bandit_samples[key] for key in range(n)]
    bandit_bounds = [(1+beta)*_compute_bound(t=1,
                     eps=epsilon, delta=delta/n)]*n
    converged = False

    while not converged:
        # Sample from current best arm
        arm = np.argmax(np.array(bandit_bounds) + np.array(bandit_samples))
        bandit_samples[arm].append(np.random.normal(
                                   bandit_means[arm], bandit_scales[arm]))

        # Recompute estimate for this arm
        bandit_estimates[arm] = np.mean(bandit_samples[arm])

        converged = _check_convergence()

        return np.argmax(bandit_estimates)


def lucb():
    pass


def _compute_bound(t, eps, delta):
    num = (1+eps)*np.log(np.log((1+eps)*t+2)/delta)
    den = 2*t

    return (1+np.sqrt(eps))*np.sqrt(num/den)
