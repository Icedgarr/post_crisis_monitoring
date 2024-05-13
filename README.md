# Individualized post-crisis monitoring of psychiatric patients via Hidden Markov models

This library contains the code used to generate simulation data, build the model and generate the results for the study
presented in the manuscript: [DOI link](https://doi.org/10.3389/fdgth.2024.1322555). There are four main parts in this library:

- Data generation: The classes in the `post_crisis_monitoring/data_generation` folder are used to generate the patients and
  simulate a hospital.
- Parameter estimation: The functions in the `post_crisis_monitoring/parameter_estimation` folder are used to estimate the
  parameters of the model presented in the manuscript.
- Model evaluation: The functions in the `post_crisis_monitoring/model_evaluation` folder are used to compute the expected
  value of the crisis.
- Plot generation: The functions in the `post_crisis_monitoring/visualization` folder are used to create the plots of the
  manuscript.

The notebook located in `notebooks/generate_simulation_results.ipynb` can be used to replicate the results. Additional
details can be found in the manuscript: DOI link.
