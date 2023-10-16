from post_crisis_monitoring.utils import prob_observation_given_state, safe_division


def compute_baum_welch_values(hmm_params, observations):
    alphas = []
    betas = []
    alphas.append(alpha_0(hmm_params, observations[0]))
    betas.insert(0, {'G': 1, 'B': 1})
    horizon = len(observations)
    for i in range(1, horizon):
        alphas.append(alpha_t(alphas[-1], hmm_params, observations[i]))
        betas.insert(0, beta_t(betas[0], hmm_params, observations[horizon - i]))

    etas = []
    xis = []
    for i in range(horizon - 1):
        etas.append(eta_t(alphas[i], betas[i]))
        xis.append(xi_t_t1(alphas[i], betas[i + 1], hmm_params, observations[i + 1]))

    etas.append(eta_t(alphas[horizon - 1], betas[horizon - 1]))
    return alphas, betas, etas, xis


def alpha_0(hmm_params, observation):
    q, r, p, p_x0 = hmm_params['q'], hmm_params['r'], hmm_params['p'], hmm_params['p_x0']
    prob_obs_per_state = prob_observation_given_state(p)[observation]
    alpha = {
        'G': p_x0['G'] * prob_obs_per_state['G'],
        'B': p_x0['B'] * prob_obs_per_state['B']
    }
    return alpha


def alpha_t(prev_alpha, hmm_params, observation):
    q, r, p = hmm_params['q'], hmm_params['r'], hmm_params['p']
    prob_obs_per_state = prob_observation_given_state(p)[observation]
    alpha = {
        'G': prev_alpha['G'] * (1 - q) * prob_obs_per_state['G'] + prev_alpha['B'] * (1 - r) * prob_obs_per_state['G'],
        'B': prev_alpha['G'] * q * prob_obs_per_state['B'] + prev_alpha['B'] * r * prob_obs_per_state['B']
    }
    return alpha


def beta_t(next_beta, hmm_params, next_observation):
    q, r, p = hmm_params['q'], hmm_params['r'], hmm_params['p']
    prob_obs_per_state = prob_observation_given_state(p)[next_observation]
    beta = {
        'G': next_beta['G'] * (1 - q) * prob_obs_per_state['G'] + next_beta['B'] * q * prob_obs_per_state['B'],
        'B': next_beta['G'] * (1 - r) * prob_obs_per_state['G'] + next_beta['B'] * r * prob_obs_per_state['B']
    }
    return beta


def eta_t(alpha, beta):
    denom = alpha['G'] * beta['G'] + alpha['B'] * beta['B']
    eta = {
        'G': safe_division(alpha['G'] * beta['G'], denom),
        'B': safe_division(alpha['B'] * beta['B'], denom)
    }
    return eta


def xi_t_t1(alpha, next_beta, hmm_params, next_observation):
    q, r, p = hmm_params['q'], hmm_params['r'], hmm_params['p']
    prob_obs_per_state = prob_observation_given_state(p)[next_observation]
    denom = (alpha['G'] * next_beta['G'] * (1 - q) * prob_obs_per_state['G'] +
             alpha['B'] * next_beta['G'] * (1 - r) * prob_obs_per_state['G'] +
             alpha['G'] * next_beta['B'] * q * prob_obs_per_state['B'] +
             alpha['B'] * next_beta['B'] * r * prob_obs_per_state['B'])
    xi = {
        'GG': safe_division(alpha['G'] * next_beta['G'] * (1 - q) * prob_obs_per_state['G'], denom),
        'GB': safe_division(alpha['G'] * next_beta['B'] * q * prob_obs_per_state['B'], denom),
        'BG': safe_division(alpha['B'] * next_beta['G'] * (1 - r) * prob_obs_per_state['G'], denom),
        'BB': safe_division(alpha['B'] * next_beta['B'] * r * prob_obs_per_state['B'], denom)
    }
    return xi
