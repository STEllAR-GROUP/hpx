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

    git checkout 1.7.0

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

   cmake_minimum_required(VERSION 3.18)
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

.. _windows_installation:

How to install |hpx| on Windows
-------------------------------
Building HPX on Windows is straightforward, and has 2 major ways of execution. First is via installation with Visual Studio and CMake GUI, and second is to add all dependencies and install by yourself.

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

* Create a build folder, but do not run CMake inside the folder.
* Open up the CMake GUI. Enter the directory location in where's my source code.
  The source directory is one with the CMakeLists.txt and subdirectories.
  In the input box labelled "Where to build the binaries:",
  enter the full path to the build folder you created before. The build
  directory is one where all compiler outputs are stored, which includes object
  files and final executables.
*  CMake variable definitons tell CMake how you wish HPX to be build, via "Add Entry" on the GUI. 
  There are two required variables you need to define: ``BOOST_ROOT`` and
  ``HWLOC_ROOT`` These (``PATH``) variables need to be set to point to the root
  folder of your Boost and hwloc installations. 
  
  It is recommended to set
  the variable ``CMAKE_INSTALL_PREFIX`` as well. This determines where the |hpx|
  libraries will be built and installed. If this (``PATH``) variable is set, it
  has to refer to the directory where the built |hpx| files should be installed
  to.
* Pressing the configure button sets the project building running. When asked about which complier to use. 
  Visual Studio x64 is default, with support for 2012/2013. Note - the x32 build limits the number of threads 
  created by HPX.
* Press "Configure" again. Repeat this step until the "Generate" button becomes
  clickable (and until no variable definitions are marked in red anymore).
* Press "Generate" and  Open up the build folder, and double-click hpx.sln. An option for the build would show, Build the INSTALL target.


For more detailed information about using |cmake|_ please refer its
documentation and also the section :ref:`building_hpx`.


How to build |hpx| under Windows 10 x64 with Visual Studio 2015
...............................................................

Please install all the dependencies needed for the Visual Studio Installation from above, with the link for CMake given below.


* Download CMake V3.18.1 installer (or latest version) from `here
  <https://blog.kitware.com/cmake-3-18-1-available-for-download/>`__ alongside to the hwloc V1.11.0 (or the latest version) from `here
  <http://www.open-mpi.org/software/hwloc/v1.11/downloads/hwloc-win64-build-1.11.0.zip>`__ and the latest Boost libraries from `here
  <https://www.boost.org/users/download/>`__ and unpack them.
  
* Build the Boost DLLs and LIBs by using these commands from Command Line (or
  PowerShell). Make sure to run the commands in the folder where you've unpacked Boost, with CMD or Powershell

  .. code-block:: bash

     bootstrap.bat

  This batch file will set up everything needed to create a successful build.
  Now execute:

  .. code-block:: bash

     b2.exe link=shared variant=release,debug architecture=x86 address-model=64 threading=multi --build-type=complete install

  This command will start a (very long) build of all available Boost libraries.
  Please, be patient.

* Open CMake-GUI.exe and inout the directory location where you unpacked the source code you downloaded from
  |hpx|'s GitHub pages.
 Here's an example of CMake path settings, which point to
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

.. _fedora_installation:


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


Some common flags while building HPX
==========

HPX is built using CMake which allows us to toggle some build options alongside the standard HPX build. Below are some examples that are used for HPX.


 * BUILD_TESTING                    ON  // Test the local HPX build
 * CMAKE_BUILD_TYPE                 Release // checks the release version
 * CMAKE_DIR                        cmake-3.17 // shows local cmake version
 * CMAKE_INSTALL_PREFIX             /usr/local
 * HPX_CXX11_STD_ATOMIC_LIBRARIES   atomic 
 * HPX_PLATFORM                     native
 * HPX_WITH_APEX                    OFF // Build HPX with APEX 
 * HPX_WITH_ASYNC_MPI               OFF // Build HPX with ASYNC
 * HPX_WITH_COMPILE_ONLY_TESTS      ON // build HPX with compilation test-cases
 * HPX_WITH_DOCUMENTATION           ON // Build docs
 * HPX_WITH_DOCUMENTATION_OUTPUT   html // Docs output format
 * HPX_WITH_EXAMPLES                ON // Build exmaples
 * HPX_WITH_EXECUTABLE_PREFIX       ON // Build executables      
 * HPX_WITH_FAIL_COMPILE_TESTS      ON // Build even after failing COMPILE_ONLY_TESTS
 * HPX_WITH_GOOGLE_PERFTOOLS        OFF 
 * HPX_WITH_ITTNOTIFY               OFF
 * HPX_WITH_MALLOC                  system // Allocate the system MALLOC, is a oommon error
 * HPX_WITH_NETWORKING              ON
 * HPX_WITH_PAPI                    OFF
 * HPX_WITH_PARCELPORT_ACTION_COU   OFF
 * HPX_WITH_PARCELPORT_MPI          OFF
 * HPX_WITH_PARCELPORT_TCP          ON
 * HPX_WITH_SANITIZERS              OFF
 * HPX_WITH_TESTS                   ON
 * HPX_WITH_VALGRIND                OFF

Documentation
==========
To build the |hpx| documentation, you need recent versions of the following
packages:

- ``python3``
- ``sphinx`` (Python package)
- ``sphinx_rtd_theme`` (Python package)
- ``breathe 4.16.0`` (Python package)
- ``doxygen``

If the |python|_ dependencies are not available through your system package
manager, you can install them using the Python package manager ``pip``:

.. code-block:: bash
   pip install --user sphinx sphinx_rtd_theme breathe
You may need to set the following CMake variables to make sure CMake can
find the required dependencies.

.. option:: DOXYGEN_ROOT:PATH

   Specifies where to look for the installation of the |doxygen|_ tool.

.. option:: SPHINX_ROOT:PATH

   Specifies where to look for the installation of the |sphinx|_ tool.

.. option:: BREATHE_APIDOC_ROOT:PATH

   Specifies where to look for the installation of the |breathe|_ tool.

We assume you are comfortable in looking at Sphinx and CMake, and here's how to build the documentation.

.. code-block:: bash
  
   make docs

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
