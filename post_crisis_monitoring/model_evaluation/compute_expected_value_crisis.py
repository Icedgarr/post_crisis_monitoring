import numpy as np
import math
from post_crisis_monitoring.utils import safe_division
from post_crisis_monitoring.model_evaluation.ricatti_resolution import diff_equation_R


def compute_expected_value_crisis(p, q, r, t, last_obs):
    try:
        if last_obs == 'B':
            y1 = (1 - p * r) / (1 - p * r + r - q)
        elif last_obs == 'G':
            y1 = (1 - p * q) / (1 - p * r + r - q)
        else:
            raise Exception('Non defined observation. Last observation needs to be either G or B.')

        R = diff_equation_R(p, q, r)
        y_0 = safe_division((R - 2 * R * y1), (1 - 2 * R - y1))
        y_p = (1 + math.sqrt(1 - safe_division(4 * ((r - q) * (1 - p)), ((1 - p * r + r - q) ** 2)))) / 2
        y_n = (1 - math.sqrt(1 - safe_division(4 * ((r - q) * (1 - p)), ((1 - p * r + r - q) ** 2)))) / 2

        expected_value = 1 - (1 - p * r + r - q) * safe_division(
            ((y_0 - y_n) * y_p ** (t + 1) - (y_p - y_0) * y_n ** (t + 1)),
            ((y_0 - y_n) * y_p ** t - (y_p - y_0) * y_n ** t)
        )
    except ZeroDivisionError:
        expected_value = np.nan
    return expected_value
