..
    Copyright (c) 2015 Adrian Serio
    Copyright (c) 2015 Harris Brakmic
    Copyright (C) 2014 Thomas Heller
    Copyright (C) 2007-2013 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_build_system:

==================
|hpx| build system
==================

.. _prerequisites:

Prerequisites
=============

Supported platforms
-------------------

At this time, |hpx| supports the following platforms. Other platforms may
work, but we do not test |hpx| with other platforms, so please be warned.

.. table:: Supported Platforms for |hpx|

   ========= ================== ====================
   Name      Minimum Version    Architectures
   ========= ================== ====================
   Linux     2.6                x86-32, x86-64, k1om
   BlueGeneQ V1R2M0             PowerPC A2
   Windows   Any Windows system x86-32, x86-64
   Mac OSX   Any OSX system     x86-64
   ========= ================== ====================

Supported compilers
-------------------
The table below shows the supported compilers for |hpx|.

.. table:: Supported Compilers for |hpx|

   =================== ================== 
   Name                Minimum Version    
   =================== ================== 
   |gcc|_              7.0              
   |clang|_            7.0             
   |visual_cxx|_ (x64) 2015     
   =================== ================== 

Software and libraries
----------------------

The table below presents all the necessary prerequisites for building |hpx|.

.. table:: Software prerequisites for |hpx|

   ====================== =================== ================== 
   \                      Name                Minimum Version    
   ====================== =================== ================== 
   **Build System**       |cmake|_            3.18
   **Required Libraries** |boost|_            1.71.0
   \                      |hwloc|_            1.5
   \                      |asio|_             1.12.0
   ====================== =================== ================== 

The most important dependencies are |boost|_ and |hwloc|_. The installation of Boost 
is described in detail in Boost's `Getting Started <https://www.boost.org/more/getting_started/index.html>`_
document. A recent version of hwloc is required in order to support thread
pinning and NUMA awareness and can be found in |hwloc_downloads|_. 

|hpx| is written in 99.99% Standard C++ (the remaining 0.01% is platform
specific assembly code). As such, |hpx| is compilable with almost any standards
compliant C++ compiler. The code base takes advantage of C++ language and 
standard library features when available.

.. note::

   When building Boost using gcc, please note that it is required to specify a
   ``cxxflags=-std=c++17`` command line argument to ``b2`` (``bjam``).

.. note::

   In most configurations, |hpx| depends only on header-only Boost.
   Boost.Filesystem is required if the standard library does not support
   filesystem. The following are not needed by default, but are required in
   certain configurations: Boost.Chrono, Boost.DateTime, Boost.Log,
   Boost.LogSetup, Boost.Regex, and Boost.Thread.

Depending on the options you chose while building and installing |hpx|,
you will find that |hpx| may depend on several other libraries such as those
listed below.

.. note::

   In order to use a high speed parcelport, we currently recommend configuring
   |hpx| to use MPI so that MPI can be used for communication between different
   localities. Please set the CMake variable ``MPI_CXX_COMPILER`` to your MPI
   C++ compiler wrapper if not detected automatically.

.. list-table:: Optional software prerequisites for |hpx|

   * * Name
     * Minimum version
   * * |google_perftools|_
     * 1.7.1
   * * |jemalloc|_
     * 2.1.0
   * * |mimalloc|_
     * 1.0.0
   * * |papi|
     *

.. _getting_hpx:

Getting |hpx|
==============

Download a tarball of the latest release from |stellar_hpx_download|_ and
unpack it or clone the repository directly using ``git``:

.. code-block:: shell-session

    $ git clone https://github.com/STEllAR-GROUP/hpx.git

It is also recommended that you check out the latest stable tag:

.. code-block:: shell-session
    
    $ cd hpx
    $ git checkout 1.7.1

.. _building_hpx:

Building |hpx|
==============

.. _info:

Basic information
-----------------

The build system for |hpx| is based on |cmake|_, a cross-platform
build-generator tool which is not responsible for building the project 
but rather generates the files needed by your build tool (GNU make, Visual 
Studio, etc.) for building |hpx|. If CMake is not already installed in your 
system, you can download it and install it here: |cmake_download|_.

Once |cmake| has been run, the build process can be started. The |hpx| build
process is highly configurable through |cmake|, and various |cmake| variables
influence the build process. The build process consists of the following parts:

* The |hpx| core libraries (target ``core``): This forms the basic set of |hpx|
  libraries.
* |hpx| Examples (target ``examples``): This target is enabled by default and
  builds all |hpx| examples (disable by setting
  :option:`HPX_WITH_EXAMPLES:BOOL`\ ``=Off``). |hpx| examples are part of the
  ``all`` target and are included in the installation if enabled.
* |hpx| Tests (target ``tests``): This target builds the |hpx| test suite and is
  enabled by default (disable by setting :option:`HPX_WITH_TESTS:BOOL`
  ``=Off``). They are not built by the ``all`` target and have to be built
  separately.
* |hpx| Documentation (target ``docs``): This target builds the documentation,
  and is not enabled by default (enable by setting
  :option:`HPX_WITH_DOCUMENTATION:BOOL`\ ``=On``. For more information see
  :ref:`documentation`.

For a complete list of available |cmake| variables that influence the build of
|hpx|, see :ref:`cmake_variables`.

The variables can be used to refine the recipes that can be found at
:ref:`build_recipes` which show some basic steps on how to build |hpx| for a
specific platform.

In order to use |hpx|, only the core libraries are required. In order to use the optional
libraries, you need to specify them as link dependencies in your build (See
:ref:`creating_hpx_projects`).

As |hpx| is a modern C++ library which relies on C++17 by default. The use of
more recent standards can be opted into explicitly. If you want to force |hpx|
to use a specific C++ standard version, you can use the following CMake
variables:

* ``HPX_WITH_CXX17``: [Deprecated] C++17 is now the default C++ standard used in HPX.
* ``HPX_WITH_CXX20``: [Deprecated] In order to use the C++20 standard, it is preferable 
  to set CMAKE_CXX_STANDARD and HPX_USE_CMAKE_CXX_STANDARD to ON.

.. _build_types:

Build types
-----------

|cmake| can be configured to generate project files suitable for builds that
have enabled debugging support or for an optimized build (without debugging
support). The |cmake| variable used to set the build type is
``CMAKE_BUILD_TYPE`` (for more information see the `CMake Documentation
<https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html>`_).
Available build types are:

* **Debug**: Full debug symbols are available as well as additional assertions to
  help debugging. To enable the debug build type for the |hpx| API, the C++ Macro
  ``HPX_DEBUG`` is defined.
* **RelWithDebInfo**: Release build with debugging symbols. This is most useful
  for profiling applications
* **Release**: Release build. This disables assertions and enables default
  compiler optimizations.
* **RelMinSize**: Release build with optimizations for small binary sizes.

.. important::

   We currently don't guarantee ABI compatibility between Debug and Release
   builds. Please make sure that applications built against |hpx| use the same
   build type as you used to build |hpx|. For CMake builds, this means that
   the ``CMAKE_BUILD_TYPE`` variables have to match and for projects not using
   |cmake|_, the ``HPX_DEBUG`` macro has to be set in debug mode.

.. _build_recipes:

Platform specific build recipes
-------------------------------

.. _unix_installation:

Unix variants
.............

Once you have the source code and the dependencies and assuming all your dependencies are in paths
known to |cmake|, the following gets you started:

#. First, set up a separate build directory to configure the project:

   .. code-block:: shell-session

      $ mkdir build && cd build

#. To configure the project you have the following options:

   * To build the core |hpx| libraries and examples, and install them to your chosen location (recommended):

    .. code-block:: shell-session

        $ cmake -DCMAKE_INSTALL_PREFIX=/install/path ..

    .. tip::

       If you want to change |cmake| variables for your build, it is usually a good
       idea to start with a clean build directory to avoid configuration problems.
       It is especially important that you use a clean build directory when changing
       between ``Release`` and ``Debug`` modes.

   * To install |hpx| to the default system folders, simply leave out the ``CMAKE_INSTALL_PREFIX`` option:

    .. code-block:: shell-session

        $ cmake ..

   * If your dependencies are in custom locations, you may need to tell |cmake| where to find them by passing one or more options to |cmake| as shown below:

    .. code-block:: shell-session

        $ cmake -DBOOST_ROOT=/path/to/boost
              -DHWLOC_ROOT=/path/to/hwloc
              -DTCMALLOC_ROOT=/path/to/tcmalloc
              -DJEMALLOC_ROOT=/path/to/jemalloc
              [other CMake variable definitions]
              /path/to/source/tree

    For instance:

    .. code-block:: shell-session

        $ cmake -DBOOST_ROOT=~/packages/boost -DHWLOC_ROOT=/packages/hwloc -DCMAKE_INSTALL_PREFIX=~/packages/hpx ~/downloads/hpx_1.5.1

   * If you want to try |hpx| without using a custom allocator pass ``-DHPX_WITH_MALLOC=system`` to |cmake|:

    .. code-block:: shell-session

        $ cmake -DCMAKE_INSTALL_PREFIX=/install/path -DHPX_WITH_MALLOC=system ..

    .. note::
       Please pay special attention to the section about :option:`HPX_WITH_MALLOC:STRING` as this is crucial for getting decent performance.

   .. important::

       If you are building |hpx| for a system with more than 64 processing units,
       you must change the |cmake| variable ``HPX_WITH_MAX_CPU_COUNT`` (to a value at least as big as the
       number of (virtual) cores on your system). Note that the default value is 64.

   .. caution::

       Compiling and linking |hpx| needs a considerable amount of memory. It is
       advisable that at least 2 GB of memory per parallel process is available.

#. Once the configuration is complete, to build the project you run:

  .. code-block:: shell-session

      $ cmake --build . --target install

.. _windows_installation:

Windows
.......

.. note::

   The following build recipes are mostly user-contributed and may be outdated.
   We always welcome updated and new build recipes.

To build |hpx| under Windows 10 x64 with Visual Studio 2015:

* Download the CMake V3.18.1 installer (or latest version) from `here
  <https://blog.kitware.com/cmake-3-18-1-available-for-download/>`__
* Download the hwloc V1.11.0 (or the latest version) from `here
  <http://www.open-mpi.org/software/hwloc/v1.11/downloads/hwloc-win64-build-1.11.0.zip>`__
  and unpack it.
* Download the latest Boost libraries from `here
  <https://www.boost.org/users/download/>`__ and unpack them.
* Build the Boost DLLs and LIBs by using these commands from Command Line (or
  PowerShell). Open CMD/PowerShell inside the Boost dir and type in:

  .. code-block:: bash

     bootstrap.bat

  This batch file will set up everything needed to create a successful build.
  Now execute:

  .. code-block:: bash

     b2.exe link=shared variant=release,debug architecture=x86 address-model=64 threading=multi --build-type=complete install

  This command will start a (very long) build of all available Boost libraries.
  Please, be patient.

* Open CMake-GUI.exe and set up your source directory (input field 'Where is the
  source code') to the *base directory* of the source code you downloaded from
  |hpx|'s GitHub pages. Here's an example of CMake path settings, which point to
  the ``Documents/GitHub/hpx`` folder:

  .. _win32_cmake_settings1:

  .. figure:: ../_static/images/cmake_settings1.png

     Example CMake path settings.

  Inside 'Where is the source-code' enter the base directory of your |hpx|
  source directory (do not enter the "src" sub-directory!). Inside 'Where to
  build the binaries' you should put in the path where all the building processes
  will happen. This is important because the building machinery will do an
  "out-of-tree" build. CMake will not touch or change the original source files
  in any way. Instead, it will generate Visual Studio Solution Files, which
  will build |hpx| packages out of the |hpx| source tree.

* Set three new environment variables (in CMake, not in Windows environment):
  ``BOOST_ROOT``, ``HWLOC_ROOT``, ``CMAKE_INSTALL_PREFIX``. The meaning of
  these variables is as follows:

  * ``BOOST_ROOT`` the |hpx| root directory of the unpacked Boost headers/cpp files.
  * ``HWLOC_ROOT`` the |hpx| root directory of the unpacked Portable Hardware Locality
    files.
  * ``CMAKE_INSTALL_PREFIX`` the |hpx| root directory where the future builds of |hpx|
    should be installed.

    .. note::

       |hpx| is a very large software collection, so it is not recommended to use the
       default ``C:\Program Files\hpx``. Many users may prefer to use simpler paths *without*
       whitespace, like ``C:\bin\hpx`` or ``D:\bin\hpx`` etc.

  To insert new env-vars click on "Add Entry" and then insert the name inside
  "Name", select ``PATH`` as Type and put the path-name in the "Path" text field.
  Repeat this for the first three variables.

  This is how variable insertion will look:

  .. _win32_cmake_settings2:

  .. figure:: ../_static/images/cmake_settings2.png

     Example CMake adding entry.

  Alternatively, users could provide ``BOOST_LIBRARYDIR`` instead of
  ``BOOST_ROOT``; the difference is that ``BOOST_LIBRARYDIR`` should point to
  the subdirectory inside Boost root where all the compiled DLLs/LIBs are. For
  example, ``BOOST_LIBRARYDIR`` may point to the ``bin.v2`` subdirectory under
  the Boost rootdir. It is important to keep the meanings of these two variables
  separated from each other: ``BOOST_DIR`` points to the ROOT folder of the
  Boost library. ``BOOST_LIBRARYDIR`` points to the subdir inside the Boost root
  folder where the compiled binaries are.

* Click the 'Configure' button of CMake-GUI. You will be immediately presented with a
  small window where you can select the C++ compiler to be used within Visual
  Studio. This has been tested using the latest v14 (a.k.a C++ 2015) but older
  versions should be sufficient too. Make sure to select the 64Bit compiler.

* After the generate process has finished successfully, click the 'Generate'
  button. Now, CMake will put new VS Solution files into the BUILD folder you
  selected at the beginning.

* Open Visual Studio and load the ``HPX.sln`` from your build folder.

* Go to ``CMakePredefinedTargets`` and build the ``INSTALL`` project:

  .. _win32_vs_targets:

  .. figure:: ../_static/images/vs_targets_install.png

     Visual Studio INSTALL target.

  It will take some time to compile everything, and in the end you should see an
  output similar to this one:

  .. _win32_vs_build_output:

  .. figure:: ../_static/images/vs_build_output.png

     Visual Studio build output.

.. _tests_examples:

Tests and examples
------------------

Running tests
.............

To build the tests:

.. code-block:: shell-session

    $ cmake --build . --target tests

To control which tests to run use ``ctest``:

* To run single tests, for example a test for ``for_loop``:

.. code-block:: shell-session

    $ ctest --output-on-failure -R tests.unit.modules.algorithms.for_loop

* To run a whole group of tests:

.. code-block:: shell-session

    $ ctest --output-on-failure -R tests.unit

Running examples
................

* To build (and install) all examples invoke:

.. code-block:: shell-session

   $ cmake -DHPX_WITH_EXAMPLES=On .
   $ make examples
   $ make install

* To build the ``hello_world_1`` example run:

.. code-block:: shell-session

   $ make hello_world_1

|hpx| executables end up in the ``bin`` directory in your build directory. You
can now run ``hello_world_1`` and should see the following output:

.. code-block:: shell-session

   $ ./bin/hello_world_1
   Hello World!

You've just run an example which prints ``Hello World!`` from the |hpx| runtime.
The source for the example is in ``examples/quickstart/hello_world_1.cpp``. The
``hello_world_distributed`` example (also available in the
``examples/quickstart`` directory) is a distributed hello world program, which is
described in :ref:`examples_hello_world`. It provides a gentle introduction to
the distributed aspects of |hpx|.

.. tip::

   Most build targets in |hpx| have two names: a simple name and
   a hierarchical name corresponding to what type of example or
   test the target is. If you are developing |hpx| it is often helpful to run
   ``make help`` to get a list of available targets. For example, ``make help |
   grep hello_world`` outputs the following:

   .. code-block:: sh

      ... examples.quickstart.hello_world_2
      ... hello_world_2
      ... examples.quickstart.hello_world_1
      ... hello_world_1
      ... examples.quickstart.hello_world_distributed
      ... hello_world_distributed

   It is also possible to build, for instance, all quickstart examples using ``make
   examples.quickstart``.

.. include:: ../../generated/cmake_variables.rst
