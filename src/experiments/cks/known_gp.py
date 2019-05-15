from src.autoks.core.model_search_experiment import ModelSearchExperiment
from src.experiments.util.experiment_runner import KnownGPExperimentRunner

exp_runner = KnownGPExperimentRunner(ModelSearchExperiment.evolutionary_experiment)
exp_runner.run(additive_form=False, verbose=True, debug=True)
