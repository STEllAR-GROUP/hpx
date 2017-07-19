.. Copyright (c) 2007-2017 Louisiana State University

   Distributed under the Boost Software License, Version 1.0. (See accompanying
   file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

*****
 HPX
*****

HPX is a C++ Standard Library for Concurrency and Parallelism. It implements
all of the corresponding facilities as defined by the C++ Standard.
Additionally, in HPX we implement functionalities proposed as part of the
ongoing C++ standardization process. We also extend the C++ Standard APIs to
the distributed case.

The goal of HPX is to create a high quality, freely available, open source
implementation of a new programming model for conventional systems, such as
classic Linux based Beowulf clusters or multi-socket highly parallel SMP
nodes. At the same time, we want to have a very modular and well designed
runtime system architecture which would allow us to port our implementation
onto new computer system architectures. We want to use real world applications
to drive the development of the runtime system, coining out required
functionalities and converging onto a stable API which will provide a
smooth migration path for developers.

The API exposed by HPX is not only modelled after the interfaces defined by the
C++11/14/17 ISO standard, it also adheres to the programming guidelines used by the
Boost collection of C++ libraries. We aim to improve the scalability of today's
applications and to expose new levels of parallelism which are necessary to
take advantage of the exascale systems of the future.

****************************
What's so special about HPX?
****************************

* HPX exposes an uniform, standards-oriented API for ease of programming
  parallel and distributed applications.
* It enables programmers to write fully asynchronous code using hundreds
  of millions of threads.
* HPX provides unified syntax and semantics for local and remote operations.
* HPX makes concurrency manageable with dataflow and future based
  synchronization.
* It implements a rich set of runtime services supporting a broad range of
  use cases.
* HPX exposes a uniform, flexible, and extendable performance counter
  framework which can enable runtime adaptivity
* It is designed to solve problems conventionally considered to be
  scaling-impaired.
* HPX has been designed and developed for systems of any scale, from
  hand-held devices to very large scale systems.
* It is the first fully functional implementation of the ParalleX execution
  model.
* HPX is published under a liberal open-source license and has an open,
  active, and thriving developer community.


The documentation for the latest release of HPX (currently V1.0) can be
`found here <http://stellar.cct.lsu.edu/files/hpx-1.0.0/html/index.html>`_.
In publications this release of HPX can be cited as: |zenodo_doi|.

.. |zenodo_doi| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.556772.svg
     :target: https://doi.org/10.5281/zenodo.556772

Additionally, we regularly upload the current status of the documentation
(which is being worked on as we speak)
`here <http://stellar-group.github.io/hpx/docs/html/>`_. We also have a
single-page version of the documentation
`here <http://stellar-group.github.io/hpx/docs/html/hpx.html>`_.

If you plan to use HPX we suggest to start with the latest released version
(currently HPX V1.0) which can be
`downloaded here <http://stellar.cct.lsu.edu/downloads/>`_.

If you would like to work with the cutting edge version from this repository
we suggest following the current health status of the master branch by looking at
our `contiguous integration results website <http://rostam.cct.lsu.edu/console>`_.
While we try to keep the master branch stable and usable, sometimes new bugs
trick their way into the code base - you have been warned!

The `CircleCI <https://circleci.com/gh/STEllAR-GROUP/hpx>`_ contiguous
integration service tracks the current build status for the master branch:
|circleci_status|

.. |circleci_status| image:: https://circleci.com/gh/STEllAR-GROUP/hpx/tree/master.svg?style=svg
     :target: https://circleci.com/gh/STEllAR-GROUP/hpx/tree/master
     :alt: HPX master branch build status

The `AppVeyor <https://ci.appveyor.com/project/hkaiser/hpx>`_ contiguous 
integration tracks the status for Windows builds using the native Visual Studio 2017
toolchain: |appveyor_status|.

.. |appveyor_status| image:: https://ci.appveyor.com/api/projects/status/sd3ehemep05fhaj1/branch/master?svg=true
     :target: https://ci.appveyor.com/project/hkaiser/hpx/branch/master
     :alt: HPX master branch Windows build status

In any case, if you happen to run into problems we very much encourage and appreciate
any issue reports through the `issue tracker for this Github project
<http://github.com/STEllAR-GROUP/hpx/issues>`_.

Also, if you have any questions feel free to ask it over at
`stackoverflow <http://stackoverflow.com>`_
and tag the question with `hpx <http://stackoverflow.com/questions/tagged/hpx>`_.

We have adopted a
`code of conduct <https://github.com/STEllAR-GROUP/hpx/blob/master/.github/CODE_OF_CONDUCT.md>`_
for this project. Please refer to this document if you would like to know more
about the expectations for members of our community, with regard to how they
will behave toward each other.

********************
 Build Instructions
********************

All of HPX is distributed under the Boost Software License,
Version 1.0 (See accompanying file LICENSE_1_0.txt or an online copy available
`here <http://www.boost.org/LICENSE_1_0.txt>`_).

Linux
-----

1)  Before starting to build HPX, please read about the
    `prerequisites <http://stellar-group.github.io/hpx/docs/html/hpx/manual/build_system/prerequisites.html>`_.

2) Clone the master HPX git repository (or a stable tag)::

    git clone git://github.com/STEllAR-GROUP/hpx.git

3) Create a build directory. HPX requires an out-of-tree build. This means you
   will be unable to run CMake in the HPX source directory::

      cd hpx
      mkdir my_hpx_build
      cd my_hpx_build

4) Invoke CMake from your build directory, pointing the CMake driver to the root
   of your HPX source tree::

      cmake -DBOOST_ROOT=/your_boost_directory \
            -DHWLOC_ROOT=/your_hwloc_directory \
            [other CMake variable definitions] \
            /path/to/hpx/source/tree

   for instance::

      cmake -DBOOST_ROOT=~/packages/boost \
            -DHWLOC_ROOT=/packages/hwloc \
            -DCMAKE_INSTALL_PREFIX=~/packages/hpx \
            ~/downloads/hpx_0.9.99

5) Invoke GNU make. If you are on a machine with multiple cores (very likely),
   add the -jN flag to your make invocation, where N is the number of cores
   on your machine plus one::

      gmake -j5

6) To complete the build and install HPX::

      gmake install

   This will build and install the essential core components of HPX only. Use::

      gmake tests

   to build and run the tests and::

      gmake examples
      gmake install

   to build and install the examples.

Please refer `here <http://stellar-group.github.io/hpx/docs/html/hpx/manual/build_system/building_hpx/build_recipes.html#hpx.manual.build_system.building_hpx.build_recipes.unix_installation>`_
for more information about building HPX on a Linux system.

OS X (Mac)
----------

1)  Before starting to build HPX, please read about the
    `prerequisites <http://stellar-group.github.io/hpx/docs/html/hpx/manual/build_system/prerequisites.html>`_.

2) Clone the master HPX git repository (or a stable tag)::

    git clone git://github.com/STEllAR-GROUP/hpx.git

3) Create a build directory. HPX requires an out-of-tree build. This means you
   will be unable to run CMake in the HPX source directory::

      cd hpx
      mkdir my_hpx_build
      cd my_hpx_build

4) Invoke CMake from your build directory, pointing the CMake driver to the root
   of your HPX source tree::

      cmake -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
            -DBOOST_ROOT=/your_boost_directory    \
            [other CMake variable definitions]    \
            /path/to/hpx/source/tree

   for instance::

      cmake -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
            -DBOOST_ROOT=~/packages/boost \
            -DCMAKE_INSTALL_PREFIX=~/packages/hpx \
            ~/downloads/hpx_0.9.99

5) Invoke GNU make. If you are on a machine with multiple cores (very likely),
   add the -jN flag to your make invocation, where N is the number of cores
   on your machine plus one::

      make -j5

6) To complete the build and install HPX::

      make install

   This will build and install the essential core components of HPX only. Use::

      make tests

   to build and run the tests and::

      make examples
      make install

   to build and install the examples.

For more information and additional options, please see the corresponding
`documentation <http://stellar-group.github.io/hpx/docs/html/hpx/manual/build_system/building_hpx/build_recipes.html#hpx.manual.build_system.building_hpx.build_recipes.macos_installation>`_.

Windows
-------

1)  Before starting to build HPX, please read about the
    `prerequisites <http://stellar-group.github.io/hpx/docs/html/hpx/manual/build_system/prerequisites.html>`_.

2) Clone the master HPX git repository (or a stable tag). You can use
   TortoiseGIT, or the git client that Cygwin provides. The git repository can
   be found at::

    git://github.com/STEllAR-GROUP/hpx.git

3) Create a build folder. HPX requires an out-of-tree-build. This means that you
   will be unable to run CMake in the HPX source folder.

4) Open up the CMake GUI. In the input box labelled "Where is the source code:",
   enter the full path to the source folder. In the input box labelled
   "Where to build the binaries:", enter the full path to the build folder you
   created in step 2.

5) Add CMake variable definitions (if any) by clicking the "Add Entry" button
   and selecting type "String". Most probably you will need to at least add the
   directories where `Boost <http://www.boost.org>`_ is located as BOOST_ROOT
   and where `Hwloc <http://www.open-mpi.org/projects/hwloc/>`_ is located as
   HWLOC_ROOT.

6) Press the "Configure" button. A window will pop up asking you which compiler
   to use. Select the x64 Visual Studio 2012 compiler. Note that while it is
   possible to build HPX for x86 we don't recommend doing so as 32 bit runs are
   severely restricted by a 32 bit Windows system limitation affecting the number
   of HPX threads you can create.

7) If the "Generate" button is not clickable, press "Configure" again. Repeat
   this step until the "Generate" button becomes clickable.

8) Press "Generate".

9) Open up the build folder, and double-click hpx.sln.

10) Build the INSTALL target.

For more information, please see the corresponding
`section in the documentation <http://stellar-group.github.io/hpx/docs/html/hpx/manual/build_system/building_hpx/build_recipes.html#hpx.manual.build_system.building_hpx.build_recipes.windows_installation>`_

BlueGene/Q
----------

So far we only support BGClang for compiling HPX on the BlueGene/Q.

1)  Before starting to build HPX, please read about the
    `prerequisites <http://stellar-group.github.io/hpx/docs/html/hpx/manual/build_system/prerequisites.html>`_.

2) Check if BGClang is available on your installation. If not obtain and install a copy
   from the `BGClang trac page <https://trac.alcf.anl.gov/projects/llvm-bgq>`_

3) Build (and install) a recent version of `Hwloc <http://www.open-mpi.org/projects/hwloc/>`_
   With the following commands::

    ./configure \
          --host=powerpc64-bgq-linux \
          --prefix=$HOME/install/hwloc \
          --disable-shared \
          --enable-static \
          CPPFLAGS='-I/bgsys/drivers/ppcfloor ' \
                   '-I/bgsys/drivers/ppcfloor/spi/include/kernel/cnk/'
    make
    make install

4) Build (and install) a recent version of Boost, using BGClang::
   To build Boost with BGClang, you'll need to set up the following in your Boost
   ``~/user-config.jam`` file::

      # user-config.jam (put this file into your home directory)
      using clang
        :
        : bgclang++11
        :
        ;

   You can then use this as your build command::

        ./bootstrap.sh
        ./b2 --build-dir=/tmp/build-boost --layout=versioned toolset=clang -j12

5) Clone the master HPX git repository (or a stable tag)::

    git clone git://github.com/STEllAR-GROUP/hpx.git

6) Generate the HPX buildfiles using cmake::

    cmake -DHPX_PLATFORM=BlueGeneQ \
          -DCMAKE_TOOLCHAIN_FILE=/path/to/hpx/cmake/toolchains/BGQ.cmake \
          -DCMAKE_CXX_COMPILER=bgclang++11 \
          -DMPI_CXX_COMPILER=mpiclang++11 \
          -DHWLOC_ROOT=/path/to/hwloc/installation \
          -DBOOST_ROOT=/path/to/boost \
          -DHPX_MALLOC=system \
          /path/to/hpx

7) To complete the build and install HPX::

    make -j24
    make install

   This will build and install the essential core components of HPX only. Use::

    make -j24 examples
    make -j24 install

   to build and install the examples.

You can find more details about using HPX on a BlueGene/Q system
`here <http://stellar-group.github.io/hpx/docs/html/hpx/manual/build_system/building_hpx/build_recipes.html#hpx.manual.build_system.building_hpx.build_recipes.bgq_installation>`_.

Intel(R) Xeon/Phi
-----------------

After installing Boost and HWLOC, the build procedure is almost the same as
for how to build HPX on Unix Variants with the sole difference that you have
to enable the Xeon Phi in the CMake Build system. This is achieved by invoking
CMake in the following way::

    cmake \
         -DCMAKE_TOOLCHAIN_FILE=/path/to/hpx/cmake/toolchains/XeonPhi.cmake \
         -DBOOST_ROOT=$BOOST_ROOT \
         -DHWLOC_ROOT=$HWLOC_ROOT \
         /path/to/hpx

For more detailed information about building HPX for the Xeon/Phi please refer to
the `documentation <http://stellar-group.github.io/hpx/docs/html/hpx/manual/build_system/building_hpx/build_recipes.html#hpx.manual.build_system.building_hpx.build_recipes.intel_mic_installation>`_.


******************
 Acknowledgements
******************

We would like to acknowledge the NSF, DoE, DARPA, the Center for Computation
and Technology (CCT) at Louisiana State University, and the Department of
Computer Science 3 - Computer Architecture at the University of Erlangen
Nuremberg who fund and support our work.

We would also like to thank the following
organizations for granting us allocations of their compute resources:
LSU HPC, LONI, XSEDE, NERSC, and the Gauss Center for Supercomputing.

HPX is currently funded by

* The National Science Foundation through awards 1117470 (APX),
  1240655 (STAR), 1447831 (PXFS), and 1339782 (STORM).

  Any opinions, findings, and conclusions or
  recommendations expressed in this material are those of the author(s)
  and do not necessarily reflect the views of the National Science Foundation.

* The Department of Energy (DoE) through the award DE-SC0008714 (XPRESS).

  Neither the United States Government nor any agency thereof, nor any of
  their employees, makes any warranty, express or implied, or assumes any
  legal liability or responsibility for the accuracy, completeness, or
  usefulness of any information, apparatus, product, or process disclosed,
  or represents that its use would not infringe privately owned rights.
  Reference herein to any specific commercial product, process, or service
  by trade name, trademark, manufacturer, or otherwise does not necessarily
  constitute or imply its endorsement, recommendation, or favoring by the
  United States Government or any agency thereof. The views and opinions of
  authors expressed herein do not necessarily state or reflect those of the
  United States Government or any agency thereof.

* The Bavarian Research Foundation (Bayerische Forschungsstfitung) through
  the grant AZ-987-11.

* The European Commission's Horizon 2020 programme through the grant
  H2020-EU.1.2.2. 671603 (AllScale).

