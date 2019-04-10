from src.autoks.experiment import Experiment
from src.experiments.util.experiment_runner import KnownGPExperimentRunner

exp_runner = KnownGPExperimentRunner(Experiment.evolutionary_experiment)
exp_runner.run(additive_form=False, verbose=True, debug=True)
