RichardsSolver
========================================================================
A numerical package for solving Richards' equation
------------------------------------------------------------------------

RichardsSolver is a Firedrake based package (https://www.firedrakeproject.org/) for numerically solving Richards' equation in two- and three-dimensional domains.

Installation
============

RichardsSolver can be clones from Github via:

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

Available Solutions
===================

TracyRichardsSolution2D
-----------------------

Analytical solution for 2D Richards equation with exponential soil properties.
Provides both steady-state and transient solutions with two boundary condition types:

- Specified head on all boundaries
- No-flux on lateral boundaries

Reference: Tracy, F. T. (2006). Clean two- and three-dimensional analytical
solutions of Richards' equation for testing numerical solvers. Water Resources
Research, 42(8). https://doi.org/10.1029/2005WR004638

VauclinRichardsSolution2D
--------------------------

Reference solution for 2D infiltration benchmark with Haverkamp soil model.
This is a numerical benchmark without closed-form analytical solution, but
provides standard problem setup for comparison.

Reference: Vauclin, M., Khanji, D., & Vachaud, G. (1979). Experimental and
numerical study of a transient, two-dimensional unsaturated-saturated water
table recharge problem. Water Resources Research, 15(5), 1089-1101.

CockettRichardsSolution3D
--------------------------

Reference solution for 3D heterogeneous benchmark with Van Genuchten soil model.
Provides spatially varying material properties and standard problem setup.

Reference: Cockett, R., Heagy, L. J., & Haber, E. (2018). Efficient 3D
inversions using the Richards equation. Computers & Geosciences, 116, 91-102.

Documentation
=============

For complete API reference and examples, see the docstrings in each module:

- ``gwassess.tracy`` - Tracy 2D analytical solutions
- ``gwassess.vauclin`` - Vauclin 2D reference solution
- ``gwassess.cockett`` - Cockett 3D reference solution

License
=======

gwassess is licensed under the GNU Lesser General Public License v3 (LGPLv3).
See LICENSE.txt for details.

Contributing
============

Contributions are welcome! Please submit issues and pull requests on GitHub:
https://github.com/g-adopt/gwassess
