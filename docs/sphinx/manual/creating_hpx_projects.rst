..
    Copyright (C) 2018 Mikael Simberg
    Copyright (C) 2014 Thomas Heller
    Copyright (C) 2007-2013 Hartmut Kaiser

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _creating_hpx_projects:

=======================
Creating |hpx| projects
=======================

.. _pkgconfig:

Using HPX with pkg-config
=========================

.. _apps:

How to build |hpx| applications with pkg-config
-----------------------------------------------

After you are done installing |hpx|, you should be able to build the following
program. It prints ``Hello World!`` on the :term:`locality` you run it on.

.. literalinclude:: ../../examples/quickstart/simplest_hello_world_1.cpp
   :language: c++

Copy the text of this program into a file called hello_world.cpp.

Now, in the directory where you put hello_world.cpp, issue the following
commands (where ``$HPX_LOCATION`` is the build directory or
``CMAKE_INSTALL_PREFIX`` you used while building |hpx|):

.. code-block:: bash

   export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$HPX_LOCATION/lib/pkgconfig
   c++ -o hello_world hello_world.cpp \
      `pkg-config --cflags --libs hpx_application`\
       -lhpx_iostreams -DHPX_APPLICATION_NAME=hello_world

.. important::

   When using pkg-config with |hpx|, the pkg-config flags must go after the
   ``-o`` flag.

.. note::

   |hpx| libraries have different names in debug and release mode. If you want
   to link against a debug |hpx| library, you need to use the ``_debug`` suffix
   for the pkg-config name. That means instead of ``hpx_application`` or
   ``hpx_component`` you will have to use ``hpx_application_debug`` or
   ``hpx_component_debug`` Moreover, all referenced |hpx| components need to
   have a appended ``d`` suffix, e.g. instead of ``-lhpx_iostreams`` you will
   need to specify ``-lhpx_iostreamsd``.

.. important::

    If the |hpx| libraries are in a path that is not found by the dynamic
    linker. You need to add the path ``$HPX_LOCATION/lib`` to your linker search
    path (for example ``LD_LIBRARY_PATH`` on Linux).

To test the program, type:

.. code-block:: bash

   ./hello_world

which should print ``Hello World!`` and exit.

.. _comps:

How to build |hpx| components with pkg-config
---------------------------------------------

Let's try a more complex example involving an |hpx| component. An |hpx|
component is a class which exposes |hpx| actions. |hpx| components are compiled
into dynamically loaded modules called component libraries. Here's the source
code:

**hello_world_component.cpp**

.. literalinclude:: ../../examples/hello_world_component/hello_world_component.cpp
   :language: c++
   :lines: 7-29

**hello_world_component.hpp**

.. literalinclude:: ../../examples/hello_world_component/hello_world_component.hpp
   :language: c++
   :lines: 7-54

**hello_world_client.cpp**

.. literalinclude:: ../../examples/hello_world_component/hello_world_client.cpp
   :language: c++

Copy the three source files above into three files (called
``hello_world_component.cpp``, ``hello_world_component.hpp`` and
``hello_world_client.cpp`` respectively).

Now, in the directory where you put the files, run the following command to
build the component library. (where ``$HPX_LOCATION`` is the build directory or
``CMAKE_INSTALL_PREFIX`` you used while building |hpx|):

.. code-block:: bash

   export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$HPX_LOCATION/lib/pkgconfig
   c++ -o libhpx_hello_world.so hello_world_component.cpp \
      `pkg-config --cflags --libs hpx_component` \
       -lhpx_iostreams -DHPX_COMPONENT_NAME=hpx_hello_world

Now pick a directory in which to install your |hpx| component libraries. For
this example, we'll choose a directory named ``my_hpx_libs``:

.. code-block:: bash

   mkdir ~/my_hpx_libs
   mv libhpx_hello_world.so ~/my_hpx_libs

.. note::

   |hpx| libraries have different names in debug and release mode. If you want
   to link against a debug |hpx| library, you need to use the ``_debug`` suffix
   for the pkg-config name. That means instead of ``hpx_application`` or
   ``hpx_component`` you will have to use ``hpx_application_debug`` or
   ``hpx_component_debug``. Moreover, all referenced |hpx| components need to
   have a appended ``d`` suffix, e.g. instead of ``-lhpx_iostreams`` you will
   need to specify ``-lhpx_iostreamsd``.

.. important::

   If the |hpx| libraries are in a path that is not found by the dynamic linker.
   You need to add the path ``$HPX_LOCATION/lib`` to your linker search path
   (for example ``LD_LIBRARY_PATH`` on Linux).

Now, to build the application that uses this component (``hello_world_client.cpp``),
we do:

.. code-block:: bash

   export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$HPX_LOCATION/lib/pkgconfig
   c++ -o hello_world_client hello_world_client.cpp \
      ``pkg-config --cflags --libs hpx_application``\
       -L${HOME}/my_hpx_libs -lhpx_hello_world -lhpx_iostreams

.. important::

   When using pkg-config with |hpx|, the pkg-config flags must go after the
   ``-o`` flag.

Finally, you'll need to set your LD_LIBRARY_PATH before you can run the program.
To run the program, type:

.. code-block:: bash

   export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/my_hpx_libs"
   ./hello_world_client

which should print ``Hello HPX World!`` and exit.

.. _using_hpx_cmake:

Using |hpx| with CMake-based projects
=====================================

In Addition to the pkg-config support discussed on the previous pages, |hpx|
comes with full CMake support. In order to integrate |hpx| into your existing,
or new CMakeLists.txt you can leverage the `find_package
<https://www.cmake.org/cmake/help/latest/command/find_package.html>`_ command
integrated into CMake. Following is a Hello World component example using CMake.

Let's revisit what we have. We have three files which compose our example
application:

* ``hello_world_component.hpp``
* ``hello_world_component.cpp``
* ``hello_world_client.hpp``

The basic structure to include |hpx| into your CMakeLists.txt is shown here:

.. code-block:: cmake

   # Require a recent version of cmake
   cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

   # This project is C++ based.
   project(your_app CXX)

   # Instruct cmake to find the HPX settings
   find_package(HPX)

In order to have CMake find |hpx|, it needs to be told where to look for the
``HPXConfig.cmake`` file that is generated when HPX is built or installed, it is
used by ``find_package(HPX)`` to set up all the necessary macros needed to use
|hpx| in your project. The ways to achieve this are:

* set the ``HPX_DIR`` cmake variable to point to the directory containing the
  ``HPXConfig.cmake`` script on the command line when you invoke cmake:

  .. code-block:: bash

     cmake -DHPX_DIR=$HPX_LOCATION/lib/cmake/HPX ...

  where ``$HPX_LOCATION`` is the build directory or ``CMAKE_INSTALL_PREFIX`` you
  used when building/configuring |hpx|.

* set the ``CMAKE_PREFIX_PATH`` variable to the root directory of your |hpx|
  build or install location on the command line when you invoke cmake:

  .. code-block:: bash

     cmake -DCMAKE_PREFIX_PATH=$HPX_LOCATION ...

  the difference between ``CMAKE_PREFIX_PATH`` and ``HPX_DIR`` is that cmake
  will add common postfixes such as ``lib/cmake/<project`` to the
  ``MAKE_PREFIX_PATH`` and search in these locations too. Note that if your
  project uses |hpx| as well as other cmake managed projects, the paths to the
  locations of these multiple projects may be concatenated in the
  ``CMAKE_PREFIX_PATH``.

* The variables above may be set in the CMake GUI or curses ccmake interface
  instead of the command line.

Additionally, if you wish to require |hpx| for your project, replace the
``find_package(HPX)`` line with ``find_package(HPX REQUIRED)``.

You can check if |hpx| was successfully found with the ``HPX_FOUND`` CMake variable.

The simplest way to add the |hpx| component is to use the ``add_hpx_component``
macro and add it to the ``CMakeLists.txt`` file:

.. code-block:: cmake

   # build your application using HPX
   add_hpx_component(hello_world
       SOURCES hello_world_component.cpp
       HEADERS hello_world_component.hpp
       COMPONENT_DEPENDENCIES iostreams)

.. note::

   ``add_hpx_component`` adds a ``_component`` suffix to the target name. In the
   example above a ``hello_world_component`` target will be created.

The available options to ``add_hpx_component`` are:

* ``SOURCES``: The source files for that component
* ``HEADERS``: The header files for that component
* ``DEPENDENCIES``: Other libraries or targets this component depends on
* ``COMPONENT_DEPENDENCIES``: The components this component depends on
* ``PLUGIN``: Treat this component as a plugin-able library
* ``COMPILE_FLAGS``: Additional compiler flags
* ``LINK_FLAGS``: Additional linker flags
* ``FOLDER``: Add the headers and source files to this Source Group folder

..
   * ``SOURCE_ROOT``
   * ``HEADER_ROOT``
   * ``SOURCE_GLOB``
   * ``HEADER_GLOB``
   * ``OUTPUT_SUFFIX``
   * ``INSTALL_SUFFIX``

* ``EXCLUDE_FROM_ALL``: Do not build this component as part of the ``all`` target

..
   * ``LANGUAGE``

After adding the component, the way you add the executable is as follows:

.. code-block:: cmake

   # build your application using HPX
   add_hpx_executable(hello_world
       ESSENTIAL
       SOURCES hello_world_client.cpp
       COMPONENT_DEPENDENCIES hello_world)

.. note::

   ``add_hpx_executable`` automatically adds a ``_component`` suffix to dependencies
   specified in ``COMPONENT_DEPENDENCIES``, meaning you can directly use the name given
   when adding a component using ``add_hpx_component``.

When you configure your application, all you need to do is set the ``HPX_DIR``
variable to point to the installation of |hpx|!

.. note::

   All library targets built with |hpx| are exported and readily available to be
   used as arguments to `target_link_libraries
   <https://www.cmake.org/cmake/help/latest/command/target_link_libraries.html>`_
   in your targets. The |hpx| include directories are available with the
   ``HPX_INCLUDE_DIRS`` CMake variable.

.. _cmake_integrate_hpx:

CMake macros to integrate |hpx| into existing applications
----------------------------------------------------------

In addition to the ``add_hpx_component`` and ``add_hpx_executable`` you can use
the ``hpx_setup_target`` macro to have an already existing target to be used
with the |hpx| libraries:

.. code-block:: cmake

   hpx_setup_target(target)

Optional parameters are:

* ``EXPORT``: Adds it to the CMake export list HPXTargets
* ``INSTALL``: Generates a install rule for the target
* ``PLUGIN``: Treat this component as a plugin-able library
* ``TYPE``: The type can be: EXECUTABLE, LIBRARY or COMPONENT
* ``DEPENDENCIES``: Other libraries or targets this component depends on
* ``COMPONENT_DEPENDENCIES``: The components this component depends on
* ``COMPILE_FLAGS``: Additional compiler flags
* ``LINK_FLAGS``: Additional linker flags

..
   * ``NO_HPXINIT``
   * ``NOLIBS``
   * ``FOLDER``
   * ``NAME``
   * ``SOVERSION``
   * ``VERSION``

If you do not use CMake, you can still build against |hpx| but you should refer
to the section on :ref:`comps`.

.. note::

   Since |hpx| relies on dynamic libraries, the dynamic linker needs to know
   where to look for them. If |hpx| isn't installed into a path which is
   configured as a linker search path, external projects need to either set
   ``RPATH`` or adapt ``LD_LIBRARY_PATH`` to point to where the hpx libraries
   reside. In order to set ``RPATH``\ s, you can include ``HPX_SetFullRPATH`` in
   your project after all libraries you want to link against have been added.
   Please also consult the CMake documentation `here
   <https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling>`_.

.. _makefile:

Using HPX with Makefile
=======================

A basic project building with |hpx| is through creating makefiles. The process
of creating one can get complex depending upon the use of cmake parameter
``HPX_WITH_HPX_MAIN`` (which defaults to ON).

How to build |hpx| applications with makefile
---------------------------------------------

If |hpx| is installed correctly, you should be able to build and run a simple
hello world program. It prints ``Hello World!`` on the :term:`locality` you
run it on.

.. literalinclude:: ../../examples/quickstart/simplest_hello_world_1.cpp
   :language: c++

Copy the content of this program into a file called hello_world.cpp.

Now in the directory where you put hello_world.cpp, create a Makefile.
Add the following code:

.. code-block:: makefile
   
   CXX=(CXX)  # Add your favourite compiler here or let makefile choose default.

   CXXFLAGS=-O3 -std=c++17

   BOOST_ROOT=/path/to/boost
   HWLOC_ROOT=/path/to/hwloc
   TCMALLOC_ROOT=/path/to/tcmalloc
   HPX_ROOT=/path/to/hpx

   INCLUDE_DIRECTIVES=$(HPX_ROOT)/include $(BOOST_ROOT)/include $(HWLOC_ROOT)/include

   LIBRARY_DIRECTIVES=-L$(HPX_ROOT)/lib $(HPX_ROOT)/lib/libhpx_init.a $(HPX_ROOT)/lib/libhpx.so $(BOOST_ROOT)/lib/libboost_atomic-mt.so $(BOOST_ROOT)/lib/libboost_filesystem-mt.so $(BOOST_ROOT)/lib/libboost_program_options-mt.so $(BOOST_ROOT)/lib/libboost_regex-mt.so $(BOOST_ROOT)/lib/libboost_system-mt.so -lpthread $(TCMALLOC_ROOT)/libtcmalloc_minimal.so $(HWLOC_ROOT)/libhwloc.so -ldl -lrt

   LINK_FLAGS=$(HPX_ROOT)/lib/libhpx_wrap.a -Wl,-wrap=main  # should be left empty for HPX_WITH_HPX_MAIN=OFF

   hello_world: hello_world.o
      $(CXX) $(CXXFLAGS) -o hello_world hello_world.o $(LIBRARY_DIRECTIVES) $(LINK_FLAGS)

   hello_world.o:
      $(CXX) $(CXXFLAGS) -c -o hello_world.o hello_world.cpp $(INCLUDE_DIRECTIVES)

.. important::
   
   ``LINK_FLAGS`` should be left empty if HPX_WITH_HPX_MAIN is set to OFF.
   Boost in the above example is build with ``--layout=tagged``. Actual boost
   flags may vary on your build of boost.

To build the program, type:

.. code-block:: bash

   make

A successfull build should result in hello_world binary. To test, type:

.. code-block:: bash

   ./hello_world

How to build |hpx| components with makefile
-------------------------------------------

Let's try a more complex example involving an |hpx| component. An |hpx|
component is a class which exposes |hpx| actions. |hpx| components are compiled
into dynamically loaded modules called component libraries. Here's the source
code:

**hello_world_component.cpp**

.. literalinclude:: ../../examples/hello_world_component/hello_world_component.cpp
   :language: c++
   :lines: 7-29

**hello_world_component.hpp**

.. literalinclude:: ../../examples/hello_world_component/hello_world_component.hpp
   :language: c++
   :lines: 7-54

**hello_world_client.cpp**

.. literalinclude:: ../../examples/hello_world_component/hello_world_client.cpp
   :language: c++

Now in the directory, create a Makefile. Add the following code:

.. code-block:: makefile
   
   CXX=(CXX)  # Add your favourite compiler here or let makefile choose default.

   CXXFLAGS=-O3 -std=c++17

   BOOST_ROOT=/path/to/boost
   HWLOC_ROOT=/path/to/hwloc
   TCMALLOC_ROOT=/path/to/tcmalloc
   HPX_ROOT=/path/to/hpx

   INCLUDE_DIRECTIVES=$(HPX_ROOT)/include $(BOOST_ROOT)/include $(HWLOC_ROOT)/include

   LIBRARY_DIRECTIVES=-L$(HPX_ROOT)/lib $(HPX_ROOT)/lib/libhpx_init.a $(HPX_ROOT)/lib/libhpx.so $(BOOST_ROOT)/lib/libboost_atomic-mt.so $(BOOST_ROOT)/lib/libboost_filesystem-mt.so $(BOOST_ROOT)/lib/libboost_program_options-mt.so $(BOOST_ROOT)/lib/libboost_regex-mt.so $(BOOST_ROOT)/lib/libboost_system-mt.so -lpthread $(TCMALLOC_ROOT)/libtcmalloc_minimal.so $(HWLOC_ROOT)/libhwloc.so -ldl -lrt

   LINK_FLAGS=$(HPX_ROOT)/lib/libhpx_wrap.a -Wl,-wrap=main  # should be left empty for HPX_WITH_HPX_MAIN=OFF

   hello_world_client: libhpx_hello_world hello_world_client.o
     $(CXX) $(CXXFLAGS) -o hello_world_client $(LIBRARY_DIRECTIVES) libhpx_hello_world $(LINK_FLAGS)

   hello_world_client.o: hello_world_client.cpp
     $(CXX) $(CXXFLAGS) -o hello_world_client.o hello_world_client.cpp $(INCLUDE_DIRECTIVES)

   libhpx_hello_world: hello_world_component.o
     $(CXX) $(CXXFLAGS) -o libhpx_hello_world hello_world_component.o $(LIBRARY_DIRECTIVES)

   hello_world_component.o: hello_world_component.cpp
     $(CXX) $(CXXFLAGS) -c -o hello_world_component.o hello_world_component.cpp $(INCLUDE_DIRECTIVES)

To build the program, type:

.. code-block:: bash

   make

A successfull build should result in hello_world binary. To test, type:

.. code-block:: bash

   ./hello_world

.. note::
   
   Due to high variations in CMake flags and library dependencies, it is
   recommended to build |hpx| applications and components with pkg-config
   or CMakeLists.txt. Writing Makefile may result in broken builds if
   due care is not taken.
   pkg-config files and CMake systems are configured with CMake build of
   |hpx|. Hence, they are stable and provides with better support overall.
