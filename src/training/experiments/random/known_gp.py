from src.autoks.core.model_search_experiment import ModelSearchExperiment
from src.training.experiments.experiment_runner import KnownGPExperimentRunner

exp_runner = KnownGPExperimentRunner(ModelSearchExperiment.random_experiment)
exp_runner.run(additive_form=False)
