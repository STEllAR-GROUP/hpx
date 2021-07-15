..
    Copyright (C) 2018 Mikael Simberg

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _quickstart:

===========
Quick start
===========

This section is intended to get you to the point of running a basic |hpx|
program as quickly as possible. To that end we skip many details but instead
give you hints and links to more details along the way.

We assume that you are on a Unix system with access to reasonably recent
packages. You should have ``cmake`` and ``make`` available for the build system
(``pkg-config`` is also supported, see :ref:`pkgconfig`).

Getting |hpx|
=============

Download a tarball of the latest release from |stellar_hpx_download|_ and
unpack it or clone the repository directly using ``git``:

.. code-block:: sh

    git clone https://github.com/STEllAR-GROUP/hpx.git

It is also recommended that you check out the latest stable tag:

.. code-block:: sh

    git checkout 1.6.0

|hpx| dependencies
==================

The minimum dependencies needed to use |hpx| are |boost|_ and |hwloc|_. If these
are not available through your system package manager, see
:ref:`boost_installation` and :ref:`hwloc_installation` for instructions on how
to build them yourself. In addition to |boost| and |hwloc|, it is recommended
that you don't use the system allocator, but instead use either ``tcmalloc``
from |google_perftools|_ (default) or |jemalloc|_ for better performance. If you
would like to try |hpx| without a custom allocator at this point, you can
configure |hpx| to use the system allocator in the next step.

A full list of required and optional dependencies, including recommended
versions, is available at :ref:`prerequisites`.

Building |hpx|
==============

Once you have the source code and the dependencies, set up a separate build
directory and configure the project. Assuming all your dependencies are in paths
known to |cmake|, the following gets you started:

.. code-block:: sh

    # In the HPX source directory
    mkdir build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=/install/path ..
    make install

This will build the core |hpx| libraries and examples, and install them to your
chosen location. If you want to install |hpx| to system folders, simply leave out
the ``CMAKE_INSTALL_PREFIX`` option. This may take a while. To speed up the
process, launch more jobs by passing the ``-jN`` option to ``make``.

.. tip::

   Do not set only ``-j`` (i.e. ``-j`` without an explicit number of jobs)
   unless you have a lot of memory available on your machine.

.. tip::

   If you want to change |cmake| variables for your build, it is usually a good
   idea to start with a clean build directory to avoid configuration problems.
   It is especially important that you use a clean build directory when changing
   between ``Release`` and ``Debug`` modes.

If your dependencies are in custom locations, you may need to tell |cmake| where
to find them by passing one or more of the following options to |cmake|:

.. code-block:: sh

    -DBOOST_ROOT=/path/to/boost
    -DHWLOC_ROOT=/path/to/hwloc
    -DTCMALLOC_ROOT=/path/to/tcmalloc
    -DJEMALLOC_ROOT=/path/to/jemalloc

If you want to try |hpx| without using a custom allocator pass
``-DHPX_WITH_MALLOC=system`` to |cmake|.

.. important::

   If you are building |hpx| for a system with more than 64 processing units,
   you must change the |cmake| variables ``HPX_WITH_MORE_THAN_64_THREADS`` (to
   ``On``) and ``HPX_WITH_MAX_CPU_COUNT`` (to a value at least as big as the
   number of (virtual) cores on your system).

To build the tests, run ``make tests``. To run the tests, run either ``make test``
or use ``ctest`` for more control over which tests to run. You can run single
tests for example with ``ctest --output-on-failure -R
tests.unit.parallel.algorithms.for_loop`` or a whole group of tests with ``ctest
--output-on-failure -R tests.unit``.

If you did not run ``make install`` earlier, do so now or build the
``hello_world_1`` example by running:

.. code-block:: sh

   make hello_world_1

|hpx| executables end up in the ``bin`` directory in your build directory. You
can now run ``hello_world_1`` and should see the following output:

.. code-block:: sh

   ./bin/hello_world_1
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

Installing and building |hpx| via vcpkg
=======================================

You can download and install |hpx| using the `vcpkg <https://github.com/Microsoft/vcpkg>`
dependency manager:

.. code-block:: sh

    git clone https://github.com/Microsoft/vcpkg.git
    cd vcpkg
    ./bootstrap-vcpkg.sh
    ./vcpkg integrate install
    vcpkg install hpx

The |hpx| port in vcpkg is kept up to date by Microsoft team members and community
contributors. If the version is out of date, please `create an issue or pull request
<https://github.com/Microsoft/vcpkg>` on the vcpkg repository.

Hello, World!
=============

The following ``CMakeLists.txt`` is a minimal example of what you need in order to
build an executable using |cmake| and |hpx|:

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.13)
   project(my_hpx_project CXX)
   find_package(HPX REQUIRED)
   add_executable(my_hpx_program main.cpp)
   target_link_libraries(my_hpx_program HPX::hpx HPX::wrap_main HPX::iostreams_component)

.. note::

   You will most likely have more than one ``main.cpp`` file in your project.
   See the section on :ref:`using_hpx_cmake` for more details on how to use
   ``add_hpx_executable``.

.. note::

   ``HPX::wrap_main`` is required if you are implicitly using ``main()`` as the
   runtime entry point. See :ref:`minimal` for more information.

.. note::

   ``HPX::iostreams_component`` is optional for a minimal project but lets us
   use the |hpx| equivalent of ``std::cout``, i.e., the |hpx| :ref:`iostreams`
   functionality in our application.

Create a new project directory and a ``CMakeLists.txt`` with the contents above.
Also create a ``main.cpp`` with the contents below.

.. literalinclude:: ../examples/quickstart/hello_world_1.cpp
   :language: c++
   :start-after: //[hello_world_1_getting_started
   :end-before: //]

Then, in your project directory run the following:

.. code-block:: sh

   mkdir build && cd build
   cmake -DCMAKE_PREFIX_PATH=/path/to/hpx/installation ..
   make all
   ./my_hpx_program

The program looks almost like a regular C++ hello world with the exception of
the two includes and ``hpx::cout``. When you include ``hpx_main.hpp`` some
things will be done behind the scenes to make sure that ``main`` actually gets
launched on the |hpx| runtime. So while it looks almost the same you can now use
futures, ``async``, parallel algorithms and more which make use of the |hpx|
runtime with lightweight threads. ``hpx::cout`` is a replacement for
``std::cout`` to make sure printing never blocks a lightweight thread. You can
read more about ``hpx::cout`` in :ref:`iostreams`. If you rebuild and run your
program now, you should see the familiar ``Hello World!``:

.. code-block:: sh

    ./my_hpx_program
    Hello World!

.. note::

   You do not have to let |hpx| take over your main function like in the
   example. You can instead keep your normal main function, and define a
   separate ``hpx_main`` function which acts as the entry point to the |hpx|
   runtime. In that case you start the |hpx| runtime explicitly by calling
   ``hpx::init``:

   .. literalinclude:: ../examples/quickstart/hello_world_2.cpp
      :language: c++

   You can also use :cpp:func:`hpx::start` and :cpp:func:`hpx::stop` for a
   non-blocking alternative, or use :cpp:func:`hpx::resume` and
   :cpp:func:`hpx::suspend` if you need to combine |hpx| with other runtimes.

   See :ref:`starting_hpx` for more details on how to initialize and run the
   |hpx| runtime.

.. caution::

   When including ``hpx_main.hpp`` the user-defined ``main`` gets renamed and
   the real ``main`` function is defined by |hpx|. This means that the
   user-defined ``main`` must include a return statement, unlike the real
   ``main``. If you do not include the return statement, you may end up with
   confusing compile time errors mentioning ``user_main`` or even runtime
   errors.

Writing task-based applications
===============================

So far we haven't done anything that can't be done using the C++ standard
library. In this section we will give a short overview of what you can do with
|hpx| on a single node. The essence is to avoid global synchronization and break
up your application into small, composable tasks whose dependencies control the
flow of your application. Remember, however, that |hpx| allows you to write
distributed applications similarly to how you would write applications for a
single node (see :ref:`why_hpx` and
:ref:`writing_distributed_hpx_applications`).

If you are already familiar with ``async`` and ``future``\ s from the C++ standard
library, the same functionality is available in |hpx|.

The following terminology is essential when talking about task-based C++
programs:

* lightweight thread: Essential for good performance with task-based programs.
  Lightweight refers to smaller stacks and faster context switching compared to
  OS threads. Smaller overheads allow the program to be broken up into smaller
  tasks, which in turns helps the runtime fully utilize all processing units.

* ``async``: The most basic way of launching tasks asynchronously. Returns a
  ``future<T>``.

* ``future<T>``: Represents a value of type ``T`` that will be ready in the future.
  The value can be retrieved with ``get`` (blocking) and one can check if the
  value is ready with ``is_ready`` (non-blocking).

* ``shared_future<T>``: Same as ``future<T>`` but can be copied (similar to
  ``std::unique_ptr`` vs ``std::shared_ptr``).

* continuation: A function that is to be run after a previous task has run
  (represented by a future). ``then`` is a method of ``future<T>`` that takes a
  function to run next. Used to build up dataflow DAGs (directed acyclic
  graphs). ``shared_future``\ s help you split up nodes in the DAG and functions
  like ``when_all`` help you join nodes in the DAG.

The following example is a collection of the most commonly used functionality in
|hpx|:

.. literalinclude:: ../examples/quickstart/potpourri.cpp
   :language: c++
   :lines: 6-

Try copying the contents to your ``main.cpp`` file and look at the output. It can
be a good idea to go through the program step by step with a debugger. You can
also try changing the types or adding new arguments to functions to make sure
you can get the types to match. The type of the ``then`` method can be especially
tricky to get right (the continuation needs to take the future as an argument).

.. note::

   |hpx| programs accept command line arguments. The most important one is
   :option:`--hpx:threads`\ ``=N`` to set the number of OS threads used by
   |hpx|. |hpx| uses one thread per core by default. Play around with the
   example above and see what difference the number of threads makes on the
   ``sort`` function. See :ref:`launching_and_configuring` for more details on
   how and what options you can pass to |hpx|.

.. tip::

   The example above used the construction ``hpx::when_all(...).then(...)``. For
   convenience and performance it is a good idea to replace uses of
   ``hpx::when_all(...).then(...)`` with ``dataflow``. See
   :ref:`examples_interest_calculator` for more details on ``dataflow``.

.. tip::

   If possible, try to use the provided parallel algorithms instead of
   writing your own implementation. This can save you time and the resulting
   program is often faster.

Build specific instructions
==========
Platform specific notes
-----------------------

Some platforms require users to have special link and/or compiler flags specified to
build |hpx|. This is handled via CMake's support for different toolchains (see
`cmake-toolchains(7)
<https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html>`_ for more
information). This is also used for cross compilation.

|hpx| ships with a set of toolchains that can be used for compilation of |hpx|
itself and applications depending on |hpx|. Please see :ref:`cmake_toolchains`
for more information.

In order to enable full static linking with the libraries, the CMake variable
:option:`HPX_WITH_STATIC_LINKING:BOOL` has to be set to ``On``.

.. _debugging_core:

Debugging applications using core files
---------------------------------------

For |hpx| to generate useful core files, |hpx| has to be compiled without signal
and exception handlers
:option:`HPX_WITH_DISABLED_SIGNAL_EXCEPTION_HANDLERS:BOOL`. If this option is
not specified, the signal handlers change the application state. For example,
after a segmentation fault the stack trace will show the signal handler.
Similarly, unhandled exceptions are also caught by these handlers and the
stack trace will not point to the location where the unhandled exception was
thrown.

In general, core files are a helpful tool to inspect the state of the
application at the moment of the crash (post-mortem debugging), without the need
of attaching a debugger beforehand. This approach to debugging is especially
useful if the error cannot be reliably reproduced, as only a single crashed
application run is required to gain potentially helpful information like a
stacktrace.

To debug with core files, the operating system first has to be told to actually
write them. On most Unix systems this can be done by calling:

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

  For instance:

  .. code-block:: bash

     cmake -DBOOST_ROOT=~/packages/boost -DHWLOC_ROOT=/packages/hwloc -DCMAKE_INSTALL_PREFIX=~/packages/hpx ~/downloads/hpx_1.5.1

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

For more detailed information about using |cmake|, please refer to its documentation
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

Then, you can use one of the following for your build command:

.. code-block:: bash

   b2 --build-dir=/tmp/build-boost --layout=versioned toolset=clang install -j4

or:

.. code-block:: bash

   b2 --build-dir=/tmp/build-boost --layout=versioned toolset=clang install -j4

We verified this using Boost V1.53. If you use a different version, just
remember to replace ``/usr/local/include/boost-1_53`` with whatever prefix
you used in your installation.

Build |hpx|, finally
....................

.. code-block:: bash

   cd /path/to
   git clone https://github.com/STEllAR-GROUP/hpx.git
   mkdir build-hpx && cd build-hpx

To build with Clang, execute:

.. code-block:: bash

   cmake ../hpx \
       -DCMAKE_CXX_COMPILER=clang++ \
       -DBOOST_ROOT=/path/to/boost \
       -DHWLOC_ROOT=/path/to/hwloc \
       -DHPX_WITH_GENERIC_CONTEXT_COROUTINES=On
   make -j

For more detailed information about using |cmake|, please refer its documentation
and to the section :ref:`building_hpx`.

Alternative installation method of |hpx| on OS X (Mac)
......................................................

Alternatively, you can install a recent version of gcc as well as all
required libraries via MacPorts:

#. Install |macports|

#. Install CMake, gcc, hwloc:

   .. code-block:: bash

      sudo brew install cmake
      sudo brew install boost
      sudo brew install hwloc
      sudo brew install make

#. You may also want:

   .. code-block:: bash

      sudo brew install gperftools

#. If you need to build Boost manually (the Boost package of MacPorts is built
   with Clang, and unfortunately doesn't work with a GCC-build version of |hpx|):

   .. code-block:: bash

      wget https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.bz2
      tar xjf boost_1_69_0.tar.bz2
      pushd boost_1_69_0
      export BOOST_ROOT=$HOME/boost_1_69_0
      ./bootstrap.sh --prefix=$BOOST_DIR
      ./b2 -j8
      ./b2 -j8 install
      export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$BOOST_ROOT/lib
      popd

#. Build |hpx|:

   .. code-block:: bash

      git clone https://github.com/STEllAR-GROUP/hpx.git
      mkdir hpx-build
      pushd hpx-build
      export HPX_ROOT=$HOME/hpx
      cmake -DCMAKE_CXX_COMPILER=g++ \
          -DCMAKE_CXX_FLAGS="-Wno-unused-local-typedefs" \
          -DBOOST_ROOT=$BOOST_ROOT \
          -DHWLOC_ROOT=/opt/local \
          -DCMAKE_INSTALL_PREFIX=$HOME/hpx \
          -DHPX_WITH_GENERIC_CONTEXT_COROUTINES=On \
               $(pwd)/../hpx
      make -j8
      make -j8 install
      export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$HPX_ROOT/lib/hpx
      popd

#. Note that you need to set ``BOOST_ROOT``, ``HPX_ROOT`` and
   ``DYLD_LIBRARY_PATH`` (for both ``BOOST_ROOT`` and ``HPX_ROOT``) every time
   you configure, build, or run an |hpx| application.

#. Note that you need to set ``HPX_WITH_GENERIC_CONTEXT_COROUTINES=On`` for
   MacOS.

#. If you want to use |hpx| with MPI, you need to enable the MPI parcelport, and
   also specify the location of the MPI wrapper scripts. This can be done using
   the following command:

   .. code-block:: bash

      cmake -DHPX_WITH_PARCELPORT_MPI=ON \
           -DCMAKE_CXX_COMPILER=g++ \
           -DMPI_CXX_COMPILER=openmpic++ \
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
* Install the Boost library as explained in the section
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
  enter the full path to the source folder. The source directory is the one where
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
  folder of your Boost and hwloc installations. It is recommended to set
  the variable ``CMAKE_INSTALL_PREFIX`` as well. This determines where the |hpx|
  libraries will be built and installed. If this (``PATH``) variable is set, it
  has to refer to the directory where the built |hpx| files should be installed
  to.
* Press the "Configure" button. A window will pop up asking you which compilers
  to use. Select the Visual Studio 10 (64Bit) compiler (it usually is the
  default if available). The Visual Studio 2012 (64Bit) and Visual Studio 2013
  (64Bit) compilers are supported as well. Note that while it is possible to
  build |hpx| for x86, we don't recommend doing so as 32 bit runs are severely
  restricted by a 32 bit Windows system limitation affecting the number of |hpx|
  threads you can create.
* Press "Configure" again. Repeat this step until the "Generate" button becomes
  clickable (and until no variable definitions are marked in red anymore).
* Press "Generate".
* Open up the build folder, and double-click hpx.sln.
* Build the INSTALL target.

For more detailed information about using |cmake|_ please refer its
documentation and also the section :ref:`building_hpx`.

.. _howto_win32:

How to build |hpx| under Windows 10 x64 with Visual Studio 2015
...............................................................

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

Next steps
==========

If you haven't done so already, reading the :ref:`terminology` section will help
you get familiar with the terms used in |hpx|.

The :ref:`examples` section contains small, self-contained walkthroughs of
example |hpx| programs. The :ref:`examples_1d_stencil` example is a thorough,
realistic example starting from a single node implementation and going stepwise
to a distributed implementation.

The :ref:`manual` contains detailed information on writing, building and running
|hpx| applications.
