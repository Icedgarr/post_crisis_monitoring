import random
from functools import reduce

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from post_crisis_monitoring.data_generation.patient import Patient, DEFAULT_PARAMETERS


class Hospital:
    def __init__(self, number_patients, random_patient_generation=True,
                 patient_parameters=DEFAULT_PARAMETERS, seed=7):
        self.seed = seed
        self.number_patients = number_patients
        self.patients = self.generate_patients(random_patient_generation, patient_parameters)
        self.history = []
        self.week = 0

    def generate_history(self, number_weeks):
        self.history = reduce(lambda history, i: history + [self.iterate_and_observe_patients()],
                              range(number_weeks), self.history)

    def iterate_and_observe_patients(self):
        patients_data = self.get_patients_state()
        for patient in self.patients:
            patient.next_state()
        patients_data['week'] = self.week
        self.week += 1
        return patients_data

    def get_history(self):
        return pd.concat(self.history)

    def get_patients_state(self):
        names = []
        internal_states = []
        crises_observed = []
        states_observed = []
        for patient in self.patients:
            names.append(patient.name)
            internal_states.append(patient.state)
            crisis, state = patient.observe()
            crises_observed.append(crisis)
            states_observed.append(state)
        patients_states = pd.DataFrame({
            'patient': names,
            'internal_state': internal_states,
            'crisis': crises_observed,
            'states_observed': states_observed
        })
        return patients_states

    def generate_patients(self, random_generation, patient_parameters):
        if random_generation:
            np.random.seed(self.seed)
            p_generated_values = np.random.lognormal(mean=0, sigma=1.2, size=self.number_patients,)
            scaler = MinMaxScaler((0.02, 0.98))
            p = 1 - scaler.fit_transform(p_generated_values.reshape(-1, 1)).reshape(1, -1)[0]
            # p = np.clip(1-((p_generated_values - min(p_generated_values))/ max(p_generated_values) + 0.02), 0.02, 0.98)
            q_generated_vals = np.random.lognormal(mean=0,sigma=1,size=self.number_patients)
            q = np.clip((q_generated_vals - min(q_generated_vals)) / max(q_generated_vals) + 0.02, 0.02, 0.5)
            r_generated_values = np.random.lognormal(mean=0, sigma=0.3, size=self.number_patients)
            scaler = MinMaxScaler((0.02, 0.98))
            r = 1 - scaler.fit_transform(r_generated_values.reshape(-1, 1)).reshape(1, -1)[0]
            mask = np.random.randint(0, 10, size=self.number_patients) == 0
            r[mask] = 0
            # r = np.random.uniform(0.02, 0.8, 3000)
            patients = [self.create_patient_params(f'Pat_{i}', p[i], q[i], r[i]) for i in range(self.number_patients)]
        else:
            patients = [Patient(name=f'Pat_{i}', parameters=patient_parameters) for i in range(self.number_patients)]
        return patients

    def generate_patients_v1(self, random_generation, patient_parameters):
        if random_generation:
            patients = [self.generate_random_patient(name=f'Pat_{i}') for i in range(self.number_patients)]
        else:
            patients = [Patient(name=f'Pat_{i}', parameters=patient_parameters) for i in range(self.number_patients)]
        return patients

    def generate_random_patient(self, name):
        internal_variable_distance = np.random.uniform(low=0, high=10)
        parameters = {
            'initial_state': 0,
            'internal_variable_dict': {'G': internal_variable_distance / 2, 'B': -internal_variable_distance / 2},
            'prob_crisis_dict': {'G': random.choice([0, 0]),
                                 'B': random.choice(1 - np.geomspace(1e-2, 0.5, num=100))},
            'prob_observe_state_dict': {'G': random.choice(np.geomspace(0.01, 0.3, num=100)),
                                        'B': random.choice([0, 0])},
            'transition_variables': {
                'q_GB': random.choice(np.geomspace(0.01, 0.5, num=100)),
                'q_BB': random.choice(np.geomspace(0.2, 0.7, num=100)),
            }}
        return Patient(name=name, parameters=parameters)

    def create_patient_params(self, name, p, q, r):
        parameters = {
            'initial_state': 0,
            'internal_variable_dict': {'G': 0.5, 'B': -0.5},
            'prob_crisis_dict': {'G': 0,
                                 'B': p},
            'prob_observe_state_dict': {'G': 0,
                                        'B': 0},
            'transition_variables': {
                'q_GB': q,
                'q_BB': r,
            }}
        return Patient(name=name, parameters=parameters)