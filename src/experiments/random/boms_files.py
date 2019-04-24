from src.autoks.core.experiment import Experiment
from src.experiments.util.experiment_runner import BOMSFilesExperimentRunner

exp_runner = BOMSFilesExperimentRunner(Experiment.random_experiment)
exp_runner.run(additive_form=False, verbose=True, debug=True)
