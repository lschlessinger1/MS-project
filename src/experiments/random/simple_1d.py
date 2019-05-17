from src.autoks.core.model_search_experiment import ModelSearchExperiment
from src.experiments.util.synthetic_data import Sinosoid1Dataset

dataset = Sinosoid1Dataset(input_dim=1)
experiment = ModelSearchExperiment.random_experiment(dataset, verbose=True, debug=True)
experiment.run()
