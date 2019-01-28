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

    return bandit_set.pop()


def ucb(bandit_means, bandit_scales, epsilon=0.01, delta=0.1, beta=1,
        experiment=False):
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
    bandit_estimates = [bandit_samples[key][0] for key in range(n)]
    bandit_bounds = [(1+beta)*_compute_bound(t=1,
                     eps=epsilon, delta=delta/n)]*n
    converged = False
    arms_sampled = []
    n_iter = 1
    while not converged:
        # Sample from current best arm
        arm = np.argmax(np.array(bandit_bounds) + np.array(bandit_estimates))
        arms_sampled.append(arm)
        bandit_samples[arm].append(np.random.normal(
                                   bandit_means[arm], bandit_scales[arm]))

        # Recompute estimate for this arm
        bandit_estimates[arm] = np.mean(bandit_samples[arm])
        bandit_bounds[arm] = (1+beta)*_compute_bound(
                                                    t=len(bandit_samples[arm]),
                                                    eps=epsilon, delta=delta/n)

        if experiment:
            converged = n_iter >= 5006
        else:
            converged = _check_convergence_ucb(alpha, bandit_samples, n)
        n_iter += 1

    if experiment:
        return arms_sampled
    else:
        return np.argmax(bandit_estimates)


def lucb(bandit_means, bandit_scales, epsilon=0.01, delta=0.1):
    """
    Arguments:
        - bandit_means: List of Gaussian random variable means
        - bandit_scales : List of Gaussian random variable
                             standard deviations (same order)
        - epsilon : parameter for LIL bound
        - delta : parameter for LIL bound and confidence level
    """
    n = len(bandit_means)
    bandit_samples = {key: [np.random.normal(bandit_means[key],
                      bandit_scales[key])] for key in range(n)}
    bandit_estimates = [bandit_samples[key][0] for key in range(n)]
    bandit_bounds = [_compute_bound(t=1,
                     eps=epsilon, delta=delta/n)]*n
    converged = False

    while not converged:
        h_t = np.argmax(bandit_estimates)
        mask = np.ones(len(bandit_estimates))
        mask[h_t] = -np.inf
        l_t = np.argmax(np.array(bandit_estimates + np.array(bandit_bounds)) *
                        mask)
        # Sample arms
        bandit_samples[h_t].append(np.random.normal(
                                   bandit_means[h_t], bandit_scales[h_t]))
        bandit_samples[l_t].append(np.random.normal(
                                   bandit_means[l_t], bandit_scales[l_t]))
        # Recompute estimate for arms
        bandit_estimates[h_t] = np.mean(bandit_samples[h_t])
        bandit_bounds[h_t] = _compute_bound(t=len(bandit_samples[h_t]),
                                            eps=epsilon, delta=delta/n)
        bandit_estimates[l_t] = np.mean(bandit_samples[l_t])
        bandit_bounds[l_t] = _compute_bound(t=len(bandit_samples[l_t]),
                                            eps=epsilon, delta=delta/n)

        converged = _check_convergence_lucb(h_t, l_t,
                                            bandit_estimates, bandit_bounds)

    return np.argmax(bandit_estimates)


def _compute_bound(t, eps, delta):
    num = (1+eps)*np.log(np.log((1+eps)*t+2)/delta)
    den = 2*t

    return (1+np.sqrt(eps))*np.sqrt(num/den)


def _check_convergence_ucb(alpha, bandit_samples, n):
    num_samples = [len(bandit_samples[arm]) for arm in range(n)]
    m = np.max(num_samples)

    if m > alpha*(np.sum(num_samples) - m):
        return True
    else:
        return False


def _check_convergence_lucb(first_arm, second_arm, bandit_estimates,
                            bandit_bounds):
    if (bandit_estimates[first_arm] - bandit_bounds[first_arm] >
       bandit_estimates[second_arm] + bandit_bounds[second_arm]):
        return True
    else:
        return False

