def chooseTimeStep( h, hOld, timeStep, solverParameters, dx ):
   
    import numpy as np
    from gadopt import sqrt, assemble

    maxdt     = solverParameters["maxdt"]
    mindt     = solverParameters["mindt"]
    dtTol     = solverParameters['dtTol']

    maxchange = sqrt(assemble(((h - hOld) / (hOld*timeStep))**2 * dx))
    timeStepNew = dtTol / (maxchange + 1e-012)
    timeStepNew = np.minimum(timeStepNew, maxdt)
    timeStepNew = np.maximum(timeStepNew, mindt)
    timeStepNew = np.round(timeStepNew, 6)

    return timeStepNew