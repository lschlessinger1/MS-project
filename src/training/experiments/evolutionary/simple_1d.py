from src.autoks.core.model_search_experiment import ModelSearchExperiment
from src.datasets.synthetic.synthetic_data import Sinusoid1Dataset

dataset = Sinusoid1Dataset(input_dim=1)
experiment = ModelSearchExperiment.evolutionary_experiment(dataset)
experiment.run()
