from functools import reduce
from multiprocessing import Pool
from tqdm.auto import tqdm
import pandas as pd
from math import log

from post_crisis_monitoring.parameter_estimation.baum_welch_algorithm import compute_baum_welch_values
from post_crisis_monitoring.utils import safe_division, initial_state_probabilities

initial_hmm_params = {
    'p': 0.8,
    'q': 0.2,
    'r': 0.5,
}

initial_hmm_params['p_x0'] = initial_state_probabilities(initial_hmm_params['q'], initial_hmm_params['r'])


def get_estimated_patients_hmm_params(data, num_iter=100, tol=1e-5, initial_hmm_params=initial_hmm_params, pool=None):
    if pool is None:
        pool = Pool(16)
    estimated_patient_hmm_params = pool.starmap(run_estimate_patient_params,
                                                tqdm([(data, patient, initial_hmm_params, num_iter, tol)
                                                      for patient in data['pat_id'].unique()]))

    estimated_patient_hmm_params = pd.DataFrame(estimated_patient_hmm_params)
    return estimated_patient_hmm_params


def run_estimate_patient_params(data, patient, initial_hmm_params, num_iter=100, tol=1e-5):
    patient_observations = list(data.loc[data['pat_id'] == patient]['crisis_max'])
    estimated_patient_params, conv_iter = estimate_patient_hmm_params(patient_observations, initial_hmm_params.copy(),
                                                                      num_iter=num_iter, tol=tol)
    estimated_patient_params['iteration_convergence'] = conv_iter
    estimated_patient_params['pat_id'] = patient
    return estimated_patient_params


def estimate_patient_hmm_params(observations, initial_hmm_params, num_iter, tol):
    hmm_params = initial_hmm_params.copy()
    last_likelihood = 2
    for i in range(num_iter):
        alphas, betas, etas, xis = compute_baum_welch_values(hmm_params, observations)
        likelihood = log(alphas[-1]['G'] + alphas[-1]['B'])
        if abs(likelihood - last_likelihood) < tol:
            break
        hmm_params = update_hmm_params(etas, xis, observations)
        last_likelihood = likelihood
    return hmm_params, i


def update_hmm_params(etas, xis, observations):
    hmm_params = {}
    hmm_params['p_x0'] = etas[0]
    hmm_params['q'] = safe_division(reduce(lambda cum, xi: cum + xi['GB'], xis, 0),
                                    reduce(lambda cum, eta: cum + eta['G'], etas[:-1], 0))
    hmm_params['r'] = safe_division(reduce(lambda cum, xi: cum + xi['BB'], xis, 0),
                                    reduce(lambda cum, eta: cum + eta['B'], etas[:-1], 0))
    hmm_params['p'] = min(safe_division(
        reduce(lambda cum, eta_obs: cum + eta_obs[0]['B'] * eta_obs[1], zip(etas, observations), 0),
        reduce(lambda cum, eta: cum + eta['B'], etas, 0)), 0.95)
    return hmm_params
