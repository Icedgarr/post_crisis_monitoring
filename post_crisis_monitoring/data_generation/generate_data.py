import numpy as np

from post_crisis_monitoring.data_generation.hospital import Hospital, DEFAULT_PARAMETERS


def get_weekly_crisis(num_patients=3000, weeks=1000, seed=7):
    hospital = Hospital(num_patients, random_patient_generation=True, patient_parameters=DEFAULT_PARAMETERS, seed=seed)
    hospital.generate_history(weeks)
    data = hospital.get_history()
    return data, hospital


def get_time_since_last_event(events):
    """
    Given a series of True/False values, it returns the number of records since last True.
    It requires the series to be already ordered and be only for each user.
    """
    events = events.copy().astype(bool)
    x1 = (events == 0).cumsum()
    x2 = x1.where(events, np.nan).ffill()
    return (x1 - x2).astype(float)
