from src.autoks.core.model_search_experiment import ModelSearchExperiment
from src.datasets.synthetic.synthetic_data import Sinosoid1Dataset

dataset = Sinosoid1Dataset(input_dim=1)
experiment = ModelSearchExperiment.cks_experiment(dataset, verbose=True, debug=True)
experiment.run()
