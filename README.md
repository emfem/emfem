What is EMFEM?
==============

EMFEM is a parallel C++ program targeted at the forward modeling of 3D frequency-domain
geophysical electromagnetic method using adaptive finite element method. The main features
of EMFEM including:

- Modular design
- Support both MT and CSEM
- Support Arbitrary anisotropic media
- Use unstructured tetrahedral mesh to discretize complex
  structures including topography and bathymetry
- The mesh is automatic refined guided by a goal-oriented error estimator
- The linear system is solve by iterative method with AMS preconditioner
  to minimize the computation time and memory usage
- To futher impore the efficiency, the code is paralized using mesh decomposition method

Prerequisites
-------------

EMFEM uses several open source libraries, including:

- mpich or openmpi
- PETSc
  - [PETSc](https://www.mcs.anl.gov/petsc) is a library that supports parallel linear algebra,
    krylov solvers, preconditioners and many other things. Note that PETSc must be
    compiled with MUMPS and HYPRE.
- Metis
  - [Metis](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) is a library for 
    partitioning graphs and finite element meshes.
- Tetgen
  - [TetGen](http://tetgen.org/) is a program to generate tetrahedral meshes of
    any 3D polyhedral domains. EMFEM uses it to refine mesh adaptively.
- Eigen
  - [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) is a C++ template
    library for linear algebra.
- nanoflann
  - [nanoflann](https://github.com/jlblancoc/nanoflann) is a C++11 header-only library
    for building KD-Trees of datasets with different topologies. EMFEM uses it to find
    the vertex or cell closest to a receiver point efficiently.

The easiest way to install the dependencies of EMFEM is using [spack](https://spack.io).
All the dependencies can be installed with

    $ spack install mpi metis tetgen eigen petsc+mumps+hypre+superlu-dist nanoflann

Building
--------

Once installed all the dependencies and unpacked the source code of EMFEM into a
directory `/path/to/emfem`. Then configure, compile emfem with

    $ cd /path/to/emfem
    $ mkdir build
    $ cd build
    $ spack load mpi
    $ cmake -DCMAKE_BUILD_TYPE=Release \
            -DPETSC_DIR=$(spack location -i petsc) \
            -DTETGEN_DIR=$(spack location -i tetgen) \
            -DNANOFLANN_DIR=$(spack location -i nanoflann) \
            -DEIGEN_DIR=$(spack location -i eigen) \
            -DMETIS_DIR=$(spack location -i metis) ..
    $ make

Note that EMFEM has only been tested on Linux and macOS systems. For Windows users,
we recommend using WSL1/WSL2 or a virtual machine.

Usage
-----

To run the forward modeling process, the user needs to provide three files:
the model file, the data template file, and the configuration file. Then use the
flowing command to run emfem:

    $ mpirun -np <# of processes> ./emfem -options_file <configuration-file-name>

For more details on the format of these files, please refer to the [documentation](https://emfem.ceqin.me).

License
-------

EMFEM is distributed under MIT License. Please see the file [LICENSE](./LICENSE)
for more details.

Contributing
------------

The users are encouraged to open an issue for any questions or bugs. Pull requests for
any enhancement are also wellcomed.

Authors
-------

- Ce Qin, Henan Polytechnic University, Email: <ce.qin@hpu.edu.cn>.
- Ning Zhao, Henan Polytechnic University, Email: <zhaoning@hpu.edu.cn>.
- Xuben Wang, Chengdu University of Technology, Email: <wxb@cdut.edu.cn>.

How to cite EMFEM?
------------------

If you publish results made using EMFEM, please consider citing:

    Ce Qin, Xuben Wang, Ning Zhao, 2023. EMFEM: A parallel 3D modeling code
    for frequency-domain electromagnetic method using goal-oriented adaptive
    finite element method. Computers & Geosciences, 178:105403. DOI:10.1016/j.cageo.2023.105403.

    QIN Ce, WANG XuBen, ZHAO Ning, 2020. Research on the iterative solver
    of linear equations in three-dimensional finite element forward modeling
    for frequency domain electromagnetic method[J].
    Chinese Journal Of Geophysics (in Chinese), 63(8): 3180–3191. DOI:10.6038/cjg2020N0194.
