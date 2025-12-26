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

.. code-block:: python

    import gwassess
    import numpy as np

    # Initialize Tracy solution with soil and domain parameters
    solution = gwassess.TracyRichardsSolution2D(
        alpha=0.328,      # Exponential soil parameter [1/m]
        hr=-15.24,        # Reference pressure head [m]
        L=15.24,          # Domain size [m]
        theta_r=0.15,     # Residual water content [-]
        theta_s=0.45,     # Saturated water content [-]
        Ks=1.0e-05        # Saturated hydraulic conductivity [m/s]
    )

    # Evaluate pressure head at a point
    x, y, t = 7.62, 7.62, 1000.0
    h = solution.pressure_head_specified_head(x, y, t)
    print(f"Pressure head at ({x}, {y}) at time {t}: {h:.6f} m")

    # Or use Cartesian coordinates
    X = [7.62, 7.62]
    h = solution.pressure_head_cartesian(X, t, bc_type='specified_head')

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
