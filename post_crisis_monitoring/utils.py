from math import log


def prob_observation_given_state(p):
    prob_per_obs_and_state = {
        0: {
            'G': 1,
            'B': 1 - p
        },
        1: {
            'G': 0,
            'B': p
        }
    }
    return prob_per_obs_and_state


def safe_division(numerator, denominator):
    if numerator == 0:
        division = 0
    else:
        division = numerator / denominator
    return division


def safe_log(value, cap_value=-1000):
    if value <= 0:
        return cap_value
    else:
        return log(value)


def initial_state_probabilities(q, r):
    return {
        'G': (1 - r) / (1 + q - r),
        'B': q / (1 + q - r)
    }
