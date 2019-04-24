from src.autoks.core.experiment import Experiment
from src.experiments.util.experiment_runner import FileExperimentRunner

exp_runner = FileExperimentRunner(Experiment.boms_experiment)
exp_runner.run(additive_form=False, verbose=True, debug=True)
