# TimeDependentICG

A basic implementation of the time dependent ICG inversion problem.

To run this code, make sure you have fenics and dolfin-adjoint installed. (see .yml for a conda environment with all dependencies)

Execute forward_run.py, to run the forward model and generate synthetic observation data

Execute inverse_run.py to run the inverse model and find the parameter that indicates the tumor location.

Results will be saved in the corresponding folders and can be used with paraview.


Like the core FEniCS components and the dolfin-adjoint software, this piece of code is made  available under the GNU LGPL, version 3.
