from src.autoks.experiment import Experiment
from src.experiments.util.data_util import FileExperimentRunner

exp_runner = FileExperimentRunner(Experiment.boms_experiment)
exp_runner.run(additive_form=False, verbose=True, debug=True)
