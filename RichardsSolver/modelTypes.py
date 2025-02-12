def moistureContent( modelParameters, h, x, time):

    import firedrake as fd

    if modelParameters["modelType"] == "Haverkamp":

        alpha  = modelParameters["alpha"]
        beta   = modelParameters["beta"]
        thetaR = modelParameters["thetaR"]
        thetaS = modelParameters["thetaS"]

        theta = thetaR + alpha*(thetaS - thetaR) / (alpha + abs(h)**beta) ;
    
    elif modelParameters["modelType"] == "VanGenuchten":

        alpha  = modelParameters["alpha"]
        n      = modelParameters["n"]
        thetaR = modelParameters["thetaR"]
        thetaS = modelParameters["thetaS"]
        m = 1 - 1/n

        theta = ( thetaR + (thetaS - thetaR) / ( (1 + abs(alpha*h)**n)**m ) )

    elif modelParameters["modelType"] == "exponential":

        alpha  = modelParameters["alpha"]
        thetaR = modelParameters["thetaR"]
        thetaS = modelParameters["thetaS"]

        theta = thetaR + (thetaS - thetaR) * fd.exp(h * alpha) 

    else:

        print("Model type not recognised")

    return theta

def relativePermeability( modelParameters, h, x, time):

    import firedrake as fd
  
    if modelParameters["modelType"] == "Haverkamp":

        A     = modelParameters["A"]
        gamma = modelParameters["gamma"]
        Ks    = modelParameters["Ks"]

        K =  Ks*A / ( A + pow(abs(h), gamma) )

    elif modelParameters["modelType"] == "VanGenuchten":

        alpha  = modelParameters["alpha"]
        n      = modelParameters["n"]
        Ks     = modelParameters["Ks"]
        m = 1 - 1/n

        K =  Ks * (1 - (alpha*abs(h))**(n-1)* (1 + pow(alpha*abs(h), n))**(-m))**2 / (( 1 + pow(alpha*abs(h), n) )**(m/2))

    elif modelParameters["modelType"] == "exponential":

        alpha  = modelParameters["alpha"]
        Ks     = modelParameters["Ks"]

        K = Ks*fd.exp(h*alpha)

    return K

def waterRetention( modelParameters, h, x, time):

    import firedrake as fd
  
    if modelParameters["modelType"] == "Haverkamp":

        alpha  = modelParameters["alpha"]
        beta   = modelParameters["beta"]
        thetaR = modelParameters["thetaR"]
        thetaS = modelParameters["thetaS"]

        C = ( -fd.sign(h)*alpha*beta*(thetaS - thetaR)*pow(abs(h),beta-1) ) / ( pow(alpha + pow(abs(h),beta),2) )

    elif modelParameters["modelType"] == "VanGenuchten":

        alpha  = modelParameters["alpha"]
        n      = modelParameters["n"]
        thetaR = modelParameters["thetaR"]
        thetaS = modelParameters["thetaS"]
        m = 1 - 1/n

        C =  -(thetaS - thetaR) * n * m * h * (alpha**n) * (abs(h)**(n-2)) * (( alpha**n * abs(h)**n + 1)**(-m-1) ) 

    elif modelParameters["modelType"] == "exponential":

        alpha  = modelParameters["alpha"]
        thetaR = modelParameters["thetaR"]
        thetaS = modelParameters["thetaS"]

        C = (thetaS - thetaR) * fd.exp(h * alpha) * alpha 

    return C