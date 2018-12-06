..
    Copyright (c) 2015 Adrian Serio
    Copyright (c) 2015 Harris Brakmic
    Copyright (C) 2014 Thomas Heller
    Copyright (C) 2007-2013 Hartmut Kaiser

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_build_system:

==================
|hpx| build system
==================

The build system for |hpx| is based on |cmake|_. CMake is a cross-platform
build-generator tool. CMake does not build the project, it generates the files
needed by your build tool (GNU make, Visual Studio, etc.) for building |hpx|.

This section gives an introduction on how to use our build system to build
|hpx| and how to use |hpx| in your own projects.

CMake basics
============

|cmake|_ is a cross-platform build-generator tool. cmake does not build the
project, it generates the files needed by your build tool (gnu make, visual
studio, etc.) for building |hpx|.

in general, the hpx |cmake|_ scripts try to adhere to the general cmake policies
on how to write |cmake|_ based projects.

Basic CMake usage
-----------------

This section explains basic aspects of |cmake|, mostly for explaining those
options which you may need on your day-to-day usage.

CMake comes with extensive documentation in the form of html files and on the
cmake executable itself. Execute ``cmake --help`` for further help options.

CMake requires to know for which build tool it shall generate files (GNU make,
Visual Studio, Xcode, etc.). If not specified on the command line, it tries to
guess it based on you environment. Once identified the build tool, CMake uses
the corresponding Generator for creating files for your build tool. You can
explicitly specify the generator with the command line option ``-G "Name of the
generator"``. For knowing the available generators on your platform, execute:

.. code-block:: bash

   cmake --help

This will list the generator names at the end of the help text. Generator names
are case-sensitive. Example:

.. code-block:: bash

   cmake -G "Visual Studio 9 2008" path/to/hpx

For a given development platform there can be more than one adequate generator.
If you use Visual Studio ``"NMake Makefiles"`` is a generator you can use for
building with NMake. By default, CMake chooses the more specific generator
supported by your development environment. If you want an alternative generator,
you must tell this to CMake with the ``-G`` option.

.. _cmake_quick_start:

Quick start
-----------

We use here the command-line, non-interactive |cmake|_ interface.

#. Download and install CMake here: |cmake_download|_. Version 3.3.2 is the
   minimally required version for |hpx|.

#. Open a shell. Your development tools must be reachable from this shell
   through the ``PATH`` environment variable.

#. Create a directory for containing the build. It is not supported to build
   |hpx| on the source directory. cd to this directory:

   .. code-block:: bash

      mkdir mybuilddir
      cd mybuilddir

#. Execute this command on the shell replacing ``path/to/hpx/`` with the path to
   the root of your |hpx| source tree:

   .. code-block:: bash

      cmake path/to/hpx

CMake will detect your development environment, perform a series of tests and
will generate the files required for building |hpx|. CMake will use default
values for all build parameters. See the :ref:`cmake_variables` section for
fine-tuning your build.

This can fail if CMake can't detect your toolset, or if it thinks that the
environment is not sane enough. In this case make sure that the toolset that you
intend to use is the only one reachable from the shell and that the shell itself
is the correct one for you development environment. CMake will refuse to build
MinGW makefiles if you have a POSIX shell reachable through the ``PATH``
environment variable, for instance. You can force CMake to use various compilers
and tools. Please visit `CMake Useful Variables
<https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/Useful-Variables#Compilers-and-Tools>`_
for a detailed overview of specific |cmake|_ variables.

.. _options:

Options and variables
---------------------

Variables customize how the build will be generated. Options are boolean
variables, with possible values ``ON``/``OFF``. Options and variables are
defined on the CMake command line like this:

.. code-block:: bash

   cmake -DVARIABLE=value path/to/hpx

You can set a variable after the initial CMake invocation for changing its
value. You can also undefine a variable:

.. code-block:: bash

   cmake -UVARIABLE path/to/hpx

Variables are stored on the CMake cache. This is a file named ``CMakeCache.txt``
on the root of the build directory. Do not hand-edit it.

Variables are listed here appending its type after a colon. It is correct to
write the variable and the type on the CMake command line:

.. code-block:: bash

   cmake -DVARIABLE:TYPE=value path/to/llvm/source

CMake supports the following variable types: ``BOOL`` (options), ``STRING``
(arbitrary string), ``PATH`` (directory name), ``FILEPATH`` (file name).

.. _prerequisites:

Prerequisites
=============

Supported platforms
-------------------

At this time, |hpx| supports the following platforms. Other platforms may
work, but we do not test |hpx| with other platforms, so please be warned.

.. table:: Supported Platforms for |hpx|

   ========= =================== ================== ====================
   Name      Recommended Version Minimum Version    Architectures
   ========= =================== ================== ====================
   Linux     3.2                 2.6                x86-32, x86-64, k1om
   BlueGeneQ V1R2M0              V1R2M0             PowerPC A2
   Windows   7, Server 2008 R2   Any Windows system x86-32, x86-64
   Mac OSX                       Any OSX system     x86-64
   ========= =================== ================== ====================

Software and libraries
----------------------

In the simplest case, |hpx| depends on |boost|_ and |hwloc|_. So, before you
read further, please make sure you have a recent version of |boost|_ installed
on your target machine. |hpx| currently requires at least Boost V1.58.0 to work
properly. It may build and run with older versions, but we do not test |hpx|
with those versions, so please be warned.

Installing the Boost libraries is described in detail in Boost's own
|boost_getting_started| document. It is often possible to download the Boost
libraries using the package manager of your distribution. Please refer to the
corresponding documentation for your system for more information.

The installation of Boost is described in detail in Boost's own
|boost_getting_started| document. However, if you've never used the Boost
libraries (or even if you have), here's a quick primer:
:ref:`boost_installation`.

In addition, we require a recent version of hwloc in order to support thread
pinning and NUMA awareness. See :ref:`hwloc_installation` for instructions on
building |hwloc|.

|hpx| is written in 99.99% Standard C++ (the remaining 0.01% is platform
specific assembly code). As such |hpx| is compilable with almost any standards
compliant C++ compiler. A compiler supporting the C++11 Standard is highly
recommended. The code base takes advantage of C++11 language features when
available (move semantics, rvalue references, magic statics, etc.). This may
speed up the execution of your code significantly. We currently support the
following C++ compilers: GCC, MSVC, ICPC and clang. For the status of your
favorite compiler with |hpx| visit |hpx_buildbot|_.

.. list-table:: Software prerequisites for |hpx| on Linux systems.

   * * Name
     * Recommended version
     * Minimum version
     * Notes
   * * **Compilers**
     *
     *
     *
   * * |gcc|_
     * 4.9 or newer
     * 4.9
     *
   * * |icpc|_
     * 2014 or newer
     * 2014
     *
   * * |clang|_
     * 3.8 or newer
     * 3.8
     *
   * * **Build System**
     *
     *
     *
   * * |cmake|_
     * 3.9.0
     * 3.3.2
     * Cuda support 3.9
   * * **Required Libraries**
     *
     *
     *
   * * |boost_libraries|_
     * 1.62.0 or newer
     * 1.58.0
     *
   * * |hwloc|_
     * 1.11
     * 1.2 (Xeon Phi: 1.6)
     *

.. note::

   When compiling |hpx| using clang/libc++ on OSX platform it is advised not
   to use Boost V1.58 or V1.60.

.. note::

   When compiling with the Intel Compiler on Linux systems, we only support C++
   Standard Libraries provided by gcc 4.8 and upwards. If the ``g++`` in your
   path is older than 4.8, please specify the path of a newer ``g++`` by setting
   ``CMAKE_CXX_FLAGS='-gxx-name=/path/to/g++'`` via |cmake|_.

.. note::

   When building Boost using gcc please note that it is always a good idea to
   specify a ``cxxflags=-std=c++11`` command line argument to ``b2`` (``bjam``).
   Note however, that this is absolutely necessary when using gcc V5.2 and
   above.

.. list-table:: Software prerequisites for |hpx| on Windows systems

   * * Name
     * Recommended version
     * Minimum version
     * Notes
   * * **Compilers**
     *
     *
     *
   * * |visual_cxx|_ (x64)
     * 2015
     * 2015
     *
   * * **Build System**
     *
     *
     *
   * * |cmake|_
     * 3.9.0
     * 3.3.2
     *
   * * **Required Libraries**
     *
     *
     *
   * * |boost|_
     * 1.62.0 or newer
     * 1.58.0
     *
   * * |hwloc|_
     * 1.11
     * 1.5
     *

.. note::

   You need to build the following Boost libraries for |hpx|: 
   Boost.Filesystem, Boost.ProgramOptions, Boost.Regex, and Boost.System. The
   following are not needed by default, but are required in certain
   configurations: Boost.Chrono, Boost.DateTime, Boost.Log, Boost.LogSetup, and
   Boost.Thread.

Depending on the options you chose while building and installing |hpx|,
you will find that |hpx| may depend on several other libraries such as those
listed below.

.. note::

   In order to use a high speed parcelport, we currently recommend configuring
   HPX to use MPI so that MPI can be used for communication between different
   localities. Please set the CMake variable ``MPI_CXX_COMPILER`` to your MPI
   C++ compiler wrapper if not detected automatically.

.. list-table:: Highly recommended optional software prerequisites for |hpx| on
   Linux systems

   * * Name
     * Recommended version
     * Minimum version
     * Notes
   * * |google_perftools|_
     * 1.7.1
     * 1.7.1
     * Used as a replacement for the system allocator, and for allocation
       diagnostics.
   * * |libunwind|_
     * 0.99
     * 0.97
     * Dependency of google-perftools on x86-64, used for stack unwinding.
   * * |openmpi|_
     * 1.10.1
     * 1.8.0
     * Can be used as a highspeed communication library backend for the
       parcelport.

.. list-table:: Optional software prerequisites for |hpx| on Linux systems

   * * Name
     * Recommended version
     * Minimum version
     * Notes
   * * |papi|
     * Used for accessing hardware performance data.
     *
     *
   * * |jemalloc|_
     * 2.1.2
     * 2.1.0
     * Used as a replacement for the system allocator.
   * * |hdf5|_
     * 1.8.7
     * 1.6.7
     * Used for data I/O in some example applications. See important note below.

.. list-table:: Optional software prerequisites for |hpx| on Windows systems

   * * Name
     * Recommended version
     * Minimum version
     * Notes
   * * |hdf5|_
     * 1.8.7
     * 1.6.7
     * Used for data I/O in some example applications. See important note below.

.. important::

   The C++ HDF5 libraries must be compiled with enabled thread safety support.
   This has to be explicitly specified while configuring the HDF5 libraries as
   it is not the default. Additionally, you must set the following environment
   variables before configuring the HDF5 libraries (this part only needs to be
   done on Linux):

   .. code-block:: bash

      export CFLAGS='-DHDatexit=""'
      export CPPFLAGS='-DHDatexit=""'

.. _documentation_prerequisites:

Documentation
-------------

To build the |hpx| documentation you need recent versions of the following
packages:

- ``python`` (2 or 3)
- ``sphinx`` (Python package)
- ``sphinx_rtd_theme`` (Python package)
- ``breathe`` (Python package)
- ``doxygen``

If the |python|_ dependencies are not available through your system package
manager you can install them using the |python|_ package manager ``pip``:

.. code-block:: bash

   pip install --user sphinx sphinx_rtd_theme breathe

You may need to set the following |cmake|_ variables to make sure |cmake|_ can
find the required dependencies.

.. option:: DOXYGEN_ROOT:PATH

   Specifies where to look for the installation of the |doxygen|_ tool.

.. option:: SPHINX_ROOT:PATH

   Specifies where to look for the installation of the |sphinx|_ tool.

.. option:: BREATHE_APIDOC_ROOT:PATH

   Specifies where to look for the installation of the |breathe|_ tool.

.. _boost_installation:

Installing Boost
----------------

.. important::

   When building Boost using gcc please note that it is always a good idea to
   specify a ``cxxflags=-std=c++11`` command line argument to ``b2`` (``bjam``).
   Note however, that this is absolutely necessary when using gcc V5.2 and
   above.

.. important::

   On Windows, depending on the installed versions of Visual Studio, you might
   also want to pass the correct toolset to the ``b2`` command depending on
   which version of the IDE you want to use. In addition, passing
   ``address-model=64`` is highly recommended. It might be also necessary to add
   command line argument ``--build-type=complete`` to the ``b2`` command on the
   Windows platform.

The easiest way to create a working Boost installation is to compile Boost from
sources yourself. This is particularly important as many high performance
resources, even if they have Boost installed, usually only provide you with an
older version of Boost. We suggest you download the most recent release of the
Boost libraries from here: |boost_downloads|_. Unpack the downloaded archive
into a directory of your choosing. We will refer to this directory a ``$BOOST``.

Building and installing the Boost binaries is simple, regardless what platform
you are on the basic instructions are as follows (with possible additional
platform-dependent command line arguments):

.. code-block:: bash

   cd $BOOST
   bootstrap --prefix=<where to install boost>
   ./b2 -j<N>
   ./b2 install

where: ``<where to install boost>`` is the directory the built binaries will be
installed to, and ``<N>`` is the number of cores to use to build the Boost
binaries.

After the above sequence of commands has been executed (this may take a while!)
you will need to specify the directory where Boost was installed as
``BOOST_ROOT`` (``<where to install boost>``) while executing cmake for |hpx| as
explained in detail in the sections :ref:`unix_installation` and
:ref:`windows_installation`.

.. _hwloc_installation:

Installing Hwloc
----------------

.. note::

   These instructions are for everything except Windows. On Windows there is no
   need to build hwloc. Instead download the latest release, extract the files,
   and set ``HWLOC_ROOT`` during cmake configuration to the directory in which
   you extracted the files.

We suggest you download the most recent release of hwloc from here:
|hwloc_downloads|_. Unpack the downloaded archive into a directory of your
choosing. We will refer to this directory as ``$HWLOC``.

To build hwloc run:

.. code-block:: bash

   cd $HWLOC
   ./configure --prefix=<where to install hwloc>
   make -j<N> install

where: ``<where to install hwloc>`` is the directory the built binaries will be
installed to, and ``<N>`` is the number of cores to use to build hwloc.

After the above sequence of commands has been executed you will need to specify
the directory where Hwloc was installed as ``HWLOC_ROOT`` (``<where to install
hwloc>``) while executing cmake for |hpx| as explained in detail in the sections
:ref:`unix_installation` and :ref:`windows_installation`.

Please see |hwloc_doc|_ for more information about Hwloc.

.. _building_hpx:

Building |hpx|
==============

.. _info:

Basic information
-----------------

Once |cmake| has been run, the build process can be started. The |hpx| build
process is highly configurable through |cmake| and various |cmake| variables
influence the build process. The build process consists of the following parts:

* The |hpx| core libraries (target core): This forms the basic set of |hpx|
  libraries. The generated targets are:

  * ``hpx``: The core |hpx| library (always enabled).
  * ``hpx_init``: The |hpx| initialization library that applications need to
    link against to define the |hpx| entry points (disabled for static builds).
  * ``hpx_wrap``: The |hpx| static library used to determine the runtime
    behavior of |hpx| code and respective entry points for ``hpx_main.h``
  * ``iostreams_component``: The component used for (distributed) IO (always
    enabled).
  * ``component_storage_component``: The component needed for migration to
    persistent storage.
  * ``unordered_component``: The component needed for a distributed
    (partitioned) hash table.
  * ``partioned_vector_component``: The component needed for a distributed
    (partitioned) vector.
  * ``memory_component``: A dynamically loaded plugin that exposed memory based
    performance counters (only available on Linux).
  * ``io_counter_component``: A dynamically loaded plugin plugin that exposes
    I/O performance counters (only available on Linux).
  * ``papi_component``: A dynamically loaded plugin that exposes PAPI
    performance counters (enabled with :option:`HPX_WITH_PAPI:BOOL`, default is
    ``Off``).

* |hpx| Examples (target ``examples``): This target is enabled by default and
  builds all |hpx| examples (disable by setting
  :option:`HPX_WITH_EXAMPLES:BOOL`\ ``=Off``). |hpx| examples are part of the
  ``all`` target and are included in the installation if enabled.
* |hpx| Tests (target ``tests``): This target builds the |hpx| test suite and is
  enabled by default (disable by setting :option:`HPX_WITH_TESTS:BOOL`
  ``=Off``). They are not built by the ``all`` target and have to be built
  separately.
* |hpx| Documentation (target ``docs``): This target builds the documentation,
  this is not enabled by default (enable by setting
  :option:`HPX_WITH_DOCUMENTATION:BOOL`\ ``=On``. For more information see
  :ref:`documentation`.

For a complete list of available |cmake| variables that influence the build of
|hpx| see :ref:`cmake_variables`.

The variables can be used to refine the recipes that can be found
:ref:`build_recipes` which show some basic steps on how to build HPX for a
specific platform.

In order to use |hpx|, only the core libraries are required (the ones marked as
optional above are truly optional). When building against |hpx|, the |cmake|_
variable ``HPX_LIBRARIES`` will contain ``hpx`` and ``hpx_init`` (for pkgconfig,
those are added to the ``Libs`` sections). In order to use the optional
libraries, you need to specify them as link dependencies in your build (See
:ref:`creating_hpx_projects`).

As |hpx| is a modern C++ Library we require a certain minimal set of features
from the C++11 standard. In addition, we make use of certain C++14 features if
the used compiler supports them. This means that the |hpx| build system will try
to determine the highest support C++ standard flavor and check for availability
of those features. That is, the default will be the highest C++ standard version
available. If you want to force |hpx| to use a specific C++ standard version you
can use the following |cmake|_ variables:

* ``HPX_WITH_CXX0X``: Enables Pre-C++11 support (This is the minimal required
  mode on older gcc versions).
* ``HPX_WITH_CXX11``: Enables C++11 support
* ``HPX_WITH_CXX14``: Enables C++14 support
* ``HPX_WITH_CXX17``: Enables C++17 support
* ``HPX_WITH_CXX2A``: Enables (experimental) C++20 support

.. _build_types:

Build types
-----------

|cmake| can be configured to generate project files suitable for builds that
have enabled debugging support or for an optimized build (without debugging
support). The |cmake| variable used to set the build type is
``CMAKE_BUILD_TYPE`` (for more information see the `CMake Documentation
<https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html>`_).
Available build types are:

* **Debug**: Full debug symbols available and additional assertions to help
  debugging. To enable the debug build type for the |hpx| API, the C++ Macro
  ``HPX_DEBUG`` is defined.
* **RelWithDebInfo**: Release build with debugging symbols. This is most useful
  for profiling applications
* **Release**: Release build. This disables assertions and enables default
  compiler optimizations.
* **RelMinSize**: Release build with optimizations for small binary sizes.

.. important::

   We currently don't guarantee ABI compatibility between Debug and Release
   builds. Please make sure that applications built against |hpx| use the same
   build type as you used to build |hpx|. For |cmake|_ builds, this means that
   the ``CMAKE_BUILD_TYPE`` variables have to match and for projects not using
   |cmake|_, the ``HPX_DEBUG`` macro has to be set in debug mode.

.. _platform:

Platform specific notes
-----------------------

Some platforms require to have special link and/or compiler flags specified to
build HPX. This is handled via |cmake|_'s support for different toolchains (see
`cmake-toolchains(7)
<https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html>`_ for more
information). This is also used for cross compilation.

|hpx| ships with a set of toolchains that can be used for compilation of |hpx|
itself and applications depending on |hpx|. Please see :ref:`cmake_toolchains`
for more information.

In order to enable full static linking with the libraries, the |cmake|_ variable
:option:`HPX_WITH_STATIC_LINKING:BOOL` has to be set to ``On``.

.. _debugging_core:

Debugging applications using core files
---------------------------------------

For |hpx| to generate useful core files, |hpx| has to be compiled without signal
and exception handlers
:option:`HPX_WITH_DISABLED_SIGNAL_EXCEPTION_HANDLERS:BOOL`. If this option is
not specified, the signal handlers change the application state. For example,
after a segmentation fault the stack trace will show the signal handler.
Similarly, unhandled exceptions are also caught by the these handlers and the
stack trace will not point to the location where the unhandled exception was
thrown.

In general, core files are a helpful tool to inspect the state of the
application at the moment of the crash (post-mortem debugging), without the need
of attaching a debugger beforehand. This approach to debugging is especially
useful if the error cannot be reliably reproduced, as only a single crashed
application run is required to gain potentially helpful information like a
stacktrace.

To debug with core files, the operating system first has to be told to actually
write them. On most unix systems this can be done by calling:

.. code-block:: bash

   ulimit -c unlimited

in the shell. Now the debugger can be started up with:

.. code-block:: bash

   gdb <application> <core file name>

The debugger should now display the last state of the application. The default
file name for core files is ``core``.

.. _build_recipes:

Platform specific build recipes
===============================

.. note::

   The following build recipes are mostly user-contributed and may be outdated.
   We always welcome updated and new build recipes.

.. _unix_installation:

How to install |hpx| on Unix variants
-------------------------------------

* Create a build directory. |hpx| requires an out-of-tree build. This means you
  will be unable to run CMake in the |hpx| source tree.

  .. code-block:: bash

     cd hpx
     mkdir my_hpx_build
     cd my_hpx_build

* Invoke CMake from your build directory, pointing the CMake driver to the root
  of your |hpx| source tree.

  .. code-block:: bash

     cmake -DBOOST_ROOT=/root/of/boost/installation \
           -DHWLOC_ROOT=/root/of/hwloc/installation
           [other CMake variable definitions] \
           /path/to/source/tree

  for instance:

  .. code-block:: bash

     cmake -DBOOST_ROOT=~/packages/boost -DHWLOC_ROOT=/packages/hwloc -DCMAKE_INSTALL_PREFIX=~/packages/hpx ~/downloads/hpx_0.9.10

* Invoke GNU make. If you are on a machine with multiple cores, add the -jN flag
  to your make invocation, where N is the number of parallel processes |hpx|
  gets compiled with.

  .. code-block:: bash

     gmake -j4

  .. caution::

     Compiling and linking |hpx| needs a considerable amount of memory. It is
     advisable that at least 2 GB of memory per parallel process is available.

  .. note::

     Many Linux distributions use ``make`` as an alias for ``gmake``.

* To complete the build and install |hpx|:

  .. code-block:: bash

     gmake install

  .. important::

     These commands will build and install the essential core components of
     |hpx| only. In order to build and run the tests, please invoke:

     .. code-block:: bash

        gmake tests && gmake test

     and in order to build (and install) all examples invoke:

     .. code-block:: bash

        cmake -DHPX_WITH_EXAMPLES=On .
        gmake examples
        gmake install

For more detailed information about using |cmake| please refer its documentation
and also the section :ref:`building_hpx`. Please pay special attention to the
section about :option:`HPX_WITH_MALLOC:STRING` as this is crucial for getting
decent performance.

.. _macos_installation:

How to install |hpx| on OS X (Mac)
----------------------------------

This section describes how to build |hpx| for OS X (Mac).

Build (and install) a recent version of Boost, using Clang and libc++
.....................................................................

To build Boost with Clang and make it link to libc++ as standard library, you'll
need to set up either of the following in your ``~/user-config.jam`` file:

.. code-block:: text

   # user-config.jam (put this file into your home directory)
   # ...

   using clang
       :
       : "/usr/bin/clang++"
       : <cxxflags>"-std=c++11 -fcolor-diagnostics"
         <linkflags>"-stdlib=libc++ -L/path/to/libcxx/lib"
       ;

(Again, remember to replace ``/path/to`` with whatever you used earlier.)

You can then use as build command either:

.. code-block:: bash

   b2 --build-dir=/tmp/build-boost --layout=versioned toolset=clang install -j4

or:

.. code-block:: bash

   b2 --build-dir=/tmp/build-boost --layout=versioned toolset=clang install -j4

We verified this using Boost V1.53. If you use a different version, just
remember to replace ``/usr/local/include/boost-1_53`` with whatever include prefix
you had in your installation.

Build HPX, finally
..................

.. code-block:: bash

   cd /path/to
   git clone https://github.com/STEllAR-GROUP/hpx.git
   mkdir build-hpx && cd build-hpx

To build with Clang 3.2, execute:

.. code-block:: bash

   cmake ../hpx \
       -DCMAKE_CXX_COMPILER=clang++ \
       -DBOOST_INCLUDE_DIR=/usr/local/include/boost-1_53 \
       -DBOOST_LIBRARY_DIR=/usr/local/lib \
       -DBOOST_SUFFIX=-clang-darwin32-mt-1_53 \
   make

To build with Clang 3.3 (trunk), execute:

.. code-block:: bash

   cmake ../hpx \
       -DCMAKE_CXX_COMPILER=clang++ \
       -DBOOST_INCLUDE_DIR=/usr/local/include/boost-1_53 \
       -DBOOST_LIBRARY_DIR=/usr/local/lib \
       -DBOOST_SUFFIX=-clang-darwin33-mt-1_53 \
   make

For more detailed information about using |cmake| please refer its documentation
and to the section :ref:`building_hpx` for.

Alternative installation method of HPX on OS X (Mac)
....................................................

Alternatively, you can install a recent version of gcc as well as all
required libraries via MacPorts:

#. Install |macports|

#. Install CMake, gcc 4.8, and hwloc:

   .. code-block:: bash

      sudo port install gcc48
      sudo port install hwloc

   You may also want:

   .. code-block:: bash

      sudo port install cmake
      sudo port install git-core

#. Make this version of gcc your default compiler:

   .. code-block:: bash

      sudo port install gcc_select
      sudo port select gcc mp-gcc48

#. Build Boost manually (the Boost package of MacPorts is built with Clang, and
   unfortunately doesn't work with a GCC-build version of HPX):

   .. code-block:: bash

      wget http://sourceforge.net/projects/boost/files/boost/1.54.0/boost_1_54_0.tar.bz2
      tar xjf boost_1_54_0.tar.bz2
      pushd boost_1_54_0
      export BOOST_ROOT=$HOME/boost_1_54_0
      ./bootstrap.sh --prefix=$BOOST_DIR
      ./b2 -j8
      ./b2 -j8 install
      export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$BOOST_ROOT/lib
      popd

#. Build HPX:

   .. code-block:: bash

      git clone https://github.com/STEllAR-GROUP/hpx.git
      mkdir hpx-build
      pushd hpx-build
      export HPX_ROOT=$HOME/hpx
      cmake -DCMAKE_C_COMPILER=gcc \
          -DCMAKE_CXX_COMPILER=g++ \
          -DCMAKE_FORTRAN_COMPILER=gfortran \
          -DCMAKE_C_FLAGS="-Wno-unused-local-typedefs" \
          -DCMAKE_CXX_FLAGS="-Wno-unused-local-typedefs" \
          -DBOOST_ROOT=$BOOST_ROOT \
          -DHWLOC_ROOT=/opt/local \
          -DCMAKE_INSTALL_PREFIX=$HOME/hpx \
               $(pwd)/../hpx
      make -j8
      make -j8 install
      export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$HPX_ROOT/lib/hpx
      popd

#. Note that you need to set ``BOOST_ROOT``, ``HPX_ROOT`` and
   ``DYLD_LIBRARY_PATH`` (for both ``BOOST_ROOT`` and ``HPX_ROOT`` every time
   you configure, build, or run an HPX application.

#. If you want to use |hpx| with MPI, you need to enable the MPI parcelport, and
   also specify the location of the MPI wrapper scripts. This can be done e.g.
   with the following command:

   .. code-block:: bash

      cmake -DHPX_WITH_PARCELPORT_MPI=ON \
           -DCMAKE_C_COMPILER=gcc \
           -DCMAKE_CXX_COMPILER=g++ \
           -DCMAKE_FORTRAN_COMPILER=gfortran \
           -DMPI_C_COMPILER=openmpicc \
           -DMPI_CXX_COMPILER=openmpic++ \
           -DMPI_FORTRAN_COMPILER=openmpif90 \
           -DCMAKE_C_FLAGS="-Wno-unused-local-typedefs" \
           -DCMAKE_CXX_FLAGS="-Wno-unused-local-typedefs" \
           -DBOOST_ROOT=$BOOST_DIR \
           -DHWLOC_ROOT=/opt/local \
           -DCMAKE_INSTALL_PREFIX=$HOME/hpx
               $(pwd)/../hpx

.. _windows_installation:

How to install |hpx| on Windows
-------------------------------

Installation of required prerequisites
......................................

* Download the Boost c++ libraries from |boost_downloads|_
* Install the boost library as explained in the section
  :ref:`boost_installation`
* Install the hwloc library as explained in the section
  :ref:`hwloc_installation`
* Download the latest version of |cmake| binaries, which are located under the
  platform section of the downloads page at |cmake_download|_.
* Download the latest version of |hpx| from the |stellar| website:
  |stellar_hpx_download|_.

Installation of the |hpx| library
.................................

* Create a build folder. |hpx| requires an out-of-tree-build. This means that
  you will be unable to run CMake in the |hpx| source folder.
* Open up the CMake GUI. In the input box labelled "Where is the source code:",
  enter the full path to the source folder. The source directory is one where
  the sources were checked out. CMakeLists.txt files in the source directory as
  well as the subdirectories describe the build to CMake. In addition to this,
  there are CMake scripts (usually ending in .cmake) stored in a special CMake
  directory. CMake does not alter any file in the source directory and doesn't
  add new ones either. In the input box labelled "Where to build the binaries:",
  enter the full path to the build folder you created before. The build
  directory is one where all compiler outputs are stored, which includes object
  files and final executables.
* Add CMake variable definitions (if any) by clicking the "Add Entry" button.
  There are two required variables you need to define: ``BOOST_ROOT`` and
  ``HWLOC_ROOT`` These (``PATH``) variables need to be set to point to the root
  folder of your |boost|_ and |hwloc|_ installations. It is recommended to set
  the variable ``CMAKE_INSTALL_PREFIX`` as well. This determines where the HPX
  libraries will be built and installed. If this (``PATH``) variable is set, it
  has to refer to the directory where the built |hpx| files should be installed
  to.
* Press the "Configure" button. A window will pop up asking you which compilers
  to use. Select the Visual Studio 10 (64Bit) compiler (it usually is the
  default if available). The Visual Studio 2012 (64Bit) and Visual Studio 2013
  (64Bit) compilers are supported as well. Note that while it is possible to
  build HPX for x86, we don't recommend doing so as 32 bit runs are severely
  restricted by a 32 bit Windows system limitation affecting the number of HPX
  threads you can create.
* Press "Configure" again. Repeat this step until the "Generate" button becomes
  clickable (and until no variable definitions are marked red anymore).
* Press "Generate".
* Open up the build folder, and double-click hpx.sln.
* Build the INSTALL target.

For more detailed information about using |cmake|_ please refer its
documentation and also the section :ref:`building_hpx`.

.. _howto_win32:

How to build HPX under Windows 10 x64 with Visual Studio 2015
.............................................................

* Download the |cmake|_ V3.4.3 installer (or latest version) from `here
  <https://blog.kitware.com/cmake-3-4-3-available-for-download/>`__
* Download the |hwloc|_ V1.11.0 (or latest version) from `here
  <http://www.open-mpi.org/software/hwloc/v1.11/downloads/hwloc-win64-build-1.11.0.zip>`__
  and unpack it.
* Download |boost|_ libraries V1.60 (or latest version) from `here
  <https://sourceforge.net/projects/boost/files/boost/1.60.0/>`__ and unpack
  them.
* Build the boost DLLs and LIBs by using these commands from Command Line (or
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
  HPX's GitHub pages. Here's an example of my CMake path settings which point to
  my ``Documents/GitHub/hpx`` folder:

  .. _win32_cmake_settings1:

  .. figure:: /_static/images/cmake_settings1.png

     Example CMake path settings.

  Inside the 'Where is the source-code' enter the base directory of your HPX
  source directory (do not enter the "src" sub-directory!) Inside 'Where to
  build the binaries' you should put in the path where all the building process
  will happen. This is important because the building machinery will do an
  "out-of-tree" build. CMake is not touching or changing in any way the original
  source files. Instead, it will generate Visual Studio Solution Files which
  will build HPX packages out of the HPX source tree.

* Set four new environment variables (in CMake, not in Windows environment, by
  the way): ``BOOST_ROOT``, ``HWLOC_ROOT``, ``CMAKE_INSTALL_PREFIX`` and
  ``HPX_WITH_BOOST_ALL_DYNAMIC_LINK``. The meaning of these variables is as
  follows:

  * ``BOOST_ROOT`` the root directory of the unpacked Boost headers/cpp files.
  * ``HWLOC_ROOT`` the root directory of the unpacked Portable Hardware Locality
    files.
  * ``CMAKE_INSTALL_PREFIX`` the "root directory" where the future builds of HPX
    should be installed to.

    .. note::

       HPX is a BIG software collection and I really don't recommend using the
       default ``C:\Program Files\hpx``. I prefer simpler paths *without* white
       space, like ``C:\bin\hpx`` or ``D:\bin\hpx`` etc.

  To insert new env-vars click on "Add Entry" and then insert the name inside
  "Name", select ``PATH`` as Type and put the path-name in "Path" text field.
  Repeat this for the first three variables.

  The last one: ``HPX_WITH_BOOST_ALL_DYNAMIC_LINK`` is a ``BOOL`` and must be
  checked (there will be a checkbox instead of a textfield).

  This is how variable insertion looks like:

  .. _win32_cmake_settings2:

  .. figure:: /_static/images/cmake_settings2.png

     Example CMake adding entry.

  Alternatively you could provide ``BOOST_LIBRARYDIR`` instead of
  ``BOOST_ROOT`` with a difference that ``BOOST_LIBRARYDIR`` should point to
  the subdirectory inside Boost root where all the compiled DLLs/LIBs are. I
  myself have used ``BOOST_LIBRARYDIR`` which pointed to the ``bin.v2``
  subdirectory under the Boost rootdir. Important is to keep the meanings of
  these two variables separated from each other: ``BOOST_DIR`` points to the
  ROOT folder of the boost library. ``BOOST_LIBRARYDIR`` points to the subdir
  inside Boost root folder where the compiled binaries are.

* Click the 'Configure' button of CMake-GUI. You will be immediately presented a
  small window where you can select the C++ compiler to be used within Visual
  Studio. In my case I have used the latest v14 (a.k.a C++ 2015) but older
  versions should be sufficient too. Make sure to select the 64Bit compiler

* After the generate process has finished successfully click the 'Generate'
  button. Now, CMake will put new VS Solution files into the BUILD folder you
  selected at the beginning.

* Open Visual Studio and load the ``HPX.sln`` from your build folder.

* Go to ``CMakePredefinedTargets`` and build the ``INSTALL`` project:

  .. _win32_vs_targets:

  .. figure:: /_static/images/vs_targets_install.png

     Visual Studio INSTALL target.

  It will take some time to compile everything and in the end you should see an
  output similar to this one:

  .. _win32_vs_build_output:

  .. figure:: /_static/images/vs_build_output.png

     Visual Studio build output.

.. _bgq_installation:

How to Install |hpx| on BlueGene/Q
----------------------------------

So far we only support BGClang for compiling |hpx| on the BlueGene/Q.

* Check if BGClang is available on your installation. If not obtain and install
  a copy from the `BGClang trac page
  <https://trac.alcf.anl.gov/projects/llvm-bgq>`_.

* Build (and install) a recent version of |hwloc_downloads|_. With the following
  commands:

  .. code-block:: bash

     ./configure \
       --host=powerpc64-bgq-linux \
       --prefix=$HOME/install/hwloc \
       --disable-shared \
       --enable-static \
       CPPFLAGS='-I/bgsys/drivers/ppcfloor -I/bgsys/drivers/ppcfloor/spi/include/kernel/cnk/'
     make
     make install

* Build (and install) a recent version of Boost, using BGClang. To build Boost
  with BGClang, you'll need to set up the following in your Boost
  ``~/user-config.jam`` file:

  .. code-block:: bash

     # user-config.jam (put this file into your home directory)
     using clang
       :
       : bgclang++11
       :
       ;

  You can then use this as your build command:

  .. code-block:: bash

     ./bootstrap.sh
     ./b2 --build-dir=/tmp/build-boost --layout=versioned toolset=clang -j12

* Clone the master |hpx| git repository (or a stable tag):

  .. code-block:: bash

     git clone git://github.com/STEllAR-GROUP/hpx.git

* Generate the |hpx| buildfiles using cmake:

  .. code-block:: bash

     cmake -DHPX_PLATFORM=BlueGeneQ \
             -DCMAKE_TOOLCHAIN_FILE=/path/to/hpx/cmake/toolchains/BGQ.cmake \
             -DCMAKE_CXX_COMPILER=bgclang++11 \
             -DMPI_CXX_COMPILER=mpiclang++11 \
             -DHWLOC_ROOT=/path/to/hwloc/installation \
             -DBOOST_ROOT=/path/to/boost \
             -DHPX_WITH_MALLOC=system \
             /path/to/hpx

* To complete the build and install |hpx|:

  .. code-block:: bash

     make -j24
     make install

  This will build and install the essential core components of |hpx| only. Use:

  .. code-block:: bash

     make -j24 examples
     make -j24 install

  to build and install the examples.

.. _intel_mic_installation:

How to Install |hpx| on the Xeon Phi
------------------------------------

Installation of the Boost Libraries
...................................

* Download |boost_downloads|_ for Linux and unpack the retrieved tarball.

* Adapt your ``~/user-config.jam`` to contain the following lines:

  .. code-block:: bash

     ## Toolset to be used for compiling for the host
     using intel
         : host
         :
         : <cxxflags>"-std=c++0x"
         ;

     ## Toolset to be used for compiling for the Xeon Phi
     using intel
         : mic
         :
         : <cxxflags>"-std=c++0x -mmic"
           <linkflags>"-std=c++0x -mmic"
         ;

* Change to the directory you unpacked boost in (from now on referred to as
  ``$BOOST_ROOT``) and execute the following commands:

  .. code-block:: bash

     ./bootstrap.sh
     ./b2 toolset=intel-mic -j<N>

  You should now have all the required boost libraries.

Installation of the Hwloc library
.................................

* Download |hwloc_downloads|_, unpack the retrieved tarball and change to the
  newly created directory.
* Run the configure-make-install procedure as follows:

  .. code-block:: bash

     CC=icc CFLAGS=-mmic CXX=icpc CXXFLAGS=-mmic LDFLAGS=-mmic ./configure --host=x86_64-k1om-linux --prefix=$HWLOC_ROOT
     make
     make install

.. important::

   The minimally required version of the |hwloc| library on the Intel Xeon Phi
   is V1.6.

You now have a working hwloc installation in ``$HWLOC_ROOT``.

Building |hpx|
..............

After all the prerequisites have been successfully installed, we can now start
building and installing |hpx|. The build procedure is almost the same as for
:ref:`unix_installation` with the sole difference that you have to enable the
Xeon Phi in the CMake Build system. This is achieved by invoking CMake in the
following way:

.. code-block:: bash

   cmake                                             \
       -DCMAKE_TOOLCHAIN_FILE=/path/to/hpx/cmake/toolchains/XeonPhi.cmake \
       -DBOOST_ROOT=$BOOST_ROOT                      \
       -DHWLOC_ROOT=$HWLOC_ROOT                      \
       /path/to/hpx

For more detailed information about using |cmake| please refer to its
documentation and to the section :ref:`building_hpx`. Please pay special
attention to the section about :option:`HPX_WITH_MALLOC:STRING` as this is
crucial for getting decent performance on the Xeon Phi.

.. _fedora_installation:

How to install |hpx| on Fedora distributions
--------------------------------------------

.. note::

   This section of the manual is based off of our collaborators Patrick Diehl's
   blog post `Installing HPX on Fedora 22
   <http://diehlpk.github.io/2015/08/04/hpx-fedora.html>`_.

* Install all packages for minimal installation:

  .. code-block:: bash

     sudo dnf install gcc-c++ cmake boost-build boost boost-devel hwloc-devel \
       hwloc gcc-gfortran papi-devel gperftools-devel docbook-dtds \
       docbook-style-xsl libsodium-devel doxygen boost-doc hdf5-devel \
       fop boost-devel boost-openmpi-devel boost-mpich-devel

* Get the development branch of HPX:

  .. code-block:: bash

     git clone https://github.com/STEllAR-GROUP/hpx.git

* Configure it with CMake:

  .. code-block:: bash

     cd hpx
     mkdir build
     cd build
     cmake -DCMAKE_INSTALL_PREFIX=/opt/hpx ..
     make -j
     make install

  .. note::

     To build HPX without examples use:

     .. code-block:: bash

        cmake -DCMAKE_INSTALL_PREFIX=/opt/hpx -DHPX_WITH_EXAMPLES=Off ..

* Add the library path of HPX to ldconfig:

  .. code-block:: bash

     sudo echo /opt/hpx/lib > /etc/ld.so.conf.d/hpx.conf
     sudo ldconfig

.. _arch_installation

How to install |hpx| on Arch distributions
------------------------------------------

* Install all packages for minimal installation:

  .. code-block:: bash

     sudo pacman -S gcc clang cmake boost hwloc gperftools

* For building the documentation you will need to further install the following:

  .. code-block:: bash

     sudo pacman -S doxygen python-pip

     pip install --user sphinx sphinx_rtd_theme breathe

The rest of the installation steps are same as provided with Fedora
or Unix variants.

.. include:: ../../generated/cmake_toolchains.rst

.. include:: ../../generated/cmake_variables.rst
