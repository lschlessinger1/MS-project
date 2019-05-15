from src.autoks.core.model_search_experiment import ModelSearchExperiment
from src.experiments.util.experiment_runner import BOMSFilesExperimentRunner

exp_runner = BOMSFilesExperimentRunner(ModelSearchExperiment.cks_experiment)
exp_runner.run(additive_form=False, verbose=True, debug=True)
