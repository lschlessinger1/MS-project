from GPy.core.parameterization import priors

RawPriorType = priors.Prior

PRIOR_DICT = dict(
    GAUSSIAN=priors.Gaussian,
    MULTIVARIATEGAUSSIAN=priors.MultivariateGaussian,
    LOGNORMAL=priors.LogGaussian,
    UNIFORM=priors.Uniform,
    GAMMA=priors.Gamma,
    INVERSEGAMMA=priors.InverseGamma,
    DGPLVMKFDA=priors.DGPLVM_KFDA,
    DGPLVMLAMBDA=priors.DGPLVM_Lamda,
    HALFT=priors.HalfT,
    EXPONENTIAL=priors.Exponential,
    STUDENTT=priors.StudentT
)

PRIOR_NAMES = list(PRIOR_DICT.keys())
