*******************************
 HPX Gyrokinetic Torodial Code 
*******************************

Gyrokinetic Torodial Code (GTC, see |nersc|_ for more information) is a
3-dimensional code which uses the Particle-In-Cell (PIC, see |utexas|_) method
to simulate microturbence in torodial fusion plasmas. PIC is a technique for
solving certain classes of partial differential equations. This application
provides a skeleton for implementing algorithms which use PIC in the HPX
programming paradigm. It reads in particles and a mesh from files and prints
where charges would be deposited. 

Options
-------

--n : std::size_t : 5
    The number of gridpoints.

--np : std::size_t : 5
    The number of particles. 

--max-num-neighbors : std::size_t : 20
    The maximum number of neighbors.

--mesh : std::string : mesh.txt
    The file containing the mesh.

--particles : std::string : particles.txt
    The file containing the particles. 

Input File Format
-----------------

printf-style format:::

    %u %lf %lf %lf 

Description::
    
    node-index x y z

Example Run
-----------

::

   gtc_client --threads 8 --num-gridpoints 64 --num-particles 150 --mesh mesh.txt --particles particle.txt

.. |nersc| replace:: NERSC GTC
.. _nersc: http://www.nersc.gov/research-and-development/benchmarking-and-workload-characterization/nersc-6-benchmarks/gtc/

.. |utexas| replace:: Introduction to PIC Codes
.. _utexas: http://farside.ph.utexas.edu/teaching/329/lectures/node96.html

