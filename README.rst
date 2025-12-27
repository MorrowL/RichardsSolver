RichardsSolver
========================================================================
A numerical package for solving Richards' equation
------------------------------------------------------------------------

RichardsSolver is a Firedrake based package (https://www.firedrakeproject.org/) for numerically solving Richards' equation in two- and three-dimensional domains.

Installation
============

This package first requires a working installation of Firedrake, with instructions found at https://www.firedrakeproject.org/install.html. RichardsSolver can be installed from Github via:

.. code-block:: bash

    pip install git+https://github.com/MorrowL/RichardsSolver.git

or to clone the whole package (inc. demos)

.. code-block:: bash

    git clone https://github.com/MorrowL/RichardsSolver.git
    pip install -e RichardsSolver

Usage
=====

Basic usage example for the Tracy 2D analytical solution:

RichardsSolver is based on Firedrake, and as such, we recommend users to familiarise themselves with a few of the tutorials available at https://www.firedrakeproject.org/documentation.html. To perform simulations, users initialise the solver via

.. code-block:: python

    eq = RichardsSolver(V=V,
                        W=W,
                        mesh=mesh,
                        soil_curves=soil_curves,
                        bcs=richards_bcs,
                        solver_parameters=solver_parameters,
                        time_integrator=time_integrator,
                        quad_degree=3)

where V is the function of pressure head, W is the vector function space for the volumetric flux (does not influence solution). For the soil curves, users have a choice of several hydrological models (HaverkampCurve, ExponentialCurve, and VanGenutchenCurve). An example of how the soil_curves dictionary is made:

.. code-block:: python

    soil_curve = HaverkampCurve(
        theta_r=0.025,         # Residual water content [-]
        theta_s=0.40,          # Saturated water content [-]
        Ks=Ks,                 # Saturated hydraulic conductivity [m/s]
        alpha=0.44,            # Fitting parameter [m]
        beta=1.2924,           # Fitting parameter [-]
        A=0.0104,              # Fitting parameter [m]
        gamma=1.5722,          # Fitting parameter [-]
        Ss=0,                  # Specific storage coefficient [1/m]
    )

Available demos
===================

Tracy's exact solution
-----------------------
A comparison with Tracy's exact solution in two- and three-dimensions with exponential soil properties. Provides comparison to steady-state and transient solutions.

Reference: Tracy, F. T. (2006). Clean two- and three-dimensional analytical
solutions of Richards' equation for testing numerical solvers. Water Resources
Research, 42(8). https://doi.org/10.1029/2005WR004638

Water table recharge
---------------------
Reproduces Vauclin's 2D test case, which involves the rechange of a water table performed with the Haverkamp soil model.

Reference: Vauclin, M., Khanji, D., & Vachaud, G. (1979). Experimental and
numerical study of a transient, two-dimensional unsaturated-saturated water
table recharge problem. Water Resources Research, 15(5), 1089-1101. https://doi.org/10.1029/WR015i005p01089

3D water infultration into heterogeneuous soil
----------------------------------------------
Reference solution for 3D heterogeneous benchmark with Van Genuchten soil model. Provides spatially varying material properties and standard problem setup.

Reference: Cockett, R., Heagy, L. J., & Haber, E. (2018). Efficient 3D
inversions using the Richards equation. Computers & Geosciences, 116, 91-102.
https://doi.org/10.1016/j.cageo.2018.04.006

Documentation
=============

For complete API reference and examples, see the docstrings in each module:

License
=======

RichardsSolver is licensed under the GNU Lesser General Public License v3 (LGPLv3).
See LICENSE.txt for details.

Contributing
============

Contributions are welcome! Please submit issues and pull requests on GitHub:
https://github.com/MorrowL/RichardsSolver/
