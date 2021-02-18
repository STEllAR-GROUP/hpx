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
