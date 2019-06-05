from src.autoks.core.model_search_experiment import ModelSearchExperiment
from src.datasets.synthetic_data import Sinosoid1Dataset

dataset = Sinosoid1Dataset(input_dim=1)
experiment = ModelSearchExperiment.evolutionary_experiment(dataset, verbose=True, debug=True)
experiment.run()
