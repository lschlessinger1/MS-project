from src.autoks.experiment import Experiment
from src.experiments.util.synthetic_data import Sinosoid1Dataset

dataset = Sinosoid1Dataset(input_dim=1)
experiment = Experiment.boms_experiment(dataset, verbose=True, debug=True)
experiment.run(title='Simple BOMS Experiment')
