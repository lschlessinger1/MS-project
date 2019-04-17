import matlab.engine
import numpy as np
from matlab.engine.matlabengine import MatlabEngine


def start_matlab():
    return matlab.engine.start_matlab()


def quit_matlab(engine):
    engine.quit()


def prob_samples_matlab(engine: MatlabEngine,
                        max_num_hyperparameters: int = 50,
                        num_samples: int = 20) -> np.ndarray:
    """Sample from low discrepancy Sobol sequence using MATLAB.

    Returns a num_samples x max_num_hyperparameters array.
    """
    engine.evalc(f"p = sobolset({max_num_hyperparameters}, 'Skip', 1e3, 'Leap', 1e2);")
    engine.evalc("p = scramble(p,'MatousekAffineOwen');")
    engine.evalc(f"probability_samples = net(p, {num_samples})")
    probability_samples = engine.eval("probability_samples")

    return np.array(probability_samples)
