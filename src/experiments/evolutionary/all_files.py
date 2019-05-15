from src.autoks.core.model_search_experiment import ModelSearchExperiment
from src.experiments.util.experiment_runner import FileExperimentRunner

exp_runner = FileExperimentRunner(ModelSearchExperiment.evolutionary_experiment)
exp_runner.run(additive_form=False, verbose=True, debug=True)
