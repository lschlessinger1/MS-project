from src.autoks.experiment import Experiment
from src.experiments.util.data_util import KnownGPExperimentRunner

exp_runner = KnownGPExperimentRunner(Experiment.boms_experiment)
exp_runner.run(additive_form=False, verbose=True, debug=True)
