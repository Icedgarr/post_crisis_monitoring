from numpy.random import binomial

DEFAULT_PARAMETERS = {
    'initial_state': None,
    'internal_variable_dict': {'G': 1, 'B': -1},
    'prob_crisis_dict': {'G': 0, 'B': 0.7},
    'prob_observe_state_dict': {'G': 0.05, 'B': 0},
    'transition_variables': {
        'q_GB': 0.2,
        'q_BB': 0.4
    }
}


class Patient:
    def __init__(self, name, parameters=DEFAULT_PARAMETERS):
        self.parameters = parameters.copy()
        self.name = name
        self.patient_possible_states = ['G', 'B']
        self.internal_variable_dict = parameters.get('internal_variable_dict')
        self.prob_observe_crisis_dict = parameters.get('prob_crisis_dict')
        self.prob_observe_state_dict = parameters.get('prob_observe_state_dict')
        self.transition_probabilities = self._initialize_transition_probabilities(
            parameters.get('transition_variables'))
        self.numerical_state = self._initialize_state(parameters.get('initial_state'))
        self.crisis_observed, self.state_observed = self._create_observation()

    @property
    def state(self):
        return self.patient_possible_states[self.numerical_state]

    def observe(self):
        return self.crisis_observed, self.state_observed

    def next_state(self):
        change_state_probability = self._get_change_state_probability(self.state)
        if binomial(1, change_state_probability) == 1:
            self._change_state()
        self.crisis_observed, self.state_observed = self._create_observation()

    def _get_change_state_probability(self, state):
        return self.transition_probabilities[state]

    def _create_observation(self):
        crisis_observed = binomial(1, self.prob_observe_crisis_dict[self.state])
        if (crisis_observed) | (binomial(1, self.prob_observe_state_dict[self.state])):
            state_observed = self.state
        else:
            state_observed = 0
        return crisis_observed, state_observed

    def _change_state(self):
        self.numerical_state = 1 - self.numerical_state

    @staticmethod
    def _initialize_state(initial_state):
        return initial_state if initial_state is not None else binomial(1, 0.5)

    @staticmethod
    def _initialize_transition_probabilities(transition_variables):
        return {
            'G': transition_variables.get('q_GB'),
            'B': 1 - transition_variables.get('q_BB')
        }

