..
   Copyright (c) 2019 The STE||AR-Group

   Distributed under the Boost Software License, Version 1.0. (See accompanying
   file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _module_structure:

================
Module structure
================

This section explains the structure of an |hpx| module.

The tool `create_library_skeleton.py
<https://github.com/STEllAR-GROUP/hpx/blob/master/libs/create_library_skeleton.py>`_
can be used to generate a basic skeleton. The structure of this skeleton is as
follows:

* ``<lib_name>/``

  * ``README.rst``
  * ``CMakeLists.txt``
  * ``cmake``
  * ``docs/``

    * ``index.rst``

  * ``examples/``

    * ``CMakeLists.txt``

  * ``include/``

    * ``hpx/``

      * ``<lib_name>``

  * ``src/``

    * ``CMakeLists.txt``

  * ``tests/``

    * ``CMakeLists.txt``
    * ``unit/``

      * ``CMakeLists.txt``

    * ``regressions/``

      * ``CMakeLists.txt``

    * ``performance/``

      * ``CMakeLists.txt``

A ``README.rst`` should be always included which explains the basic purpose of
the library and a link to the generated documentation.

The ``include`` directory should contain only headers that other libraries need.
For each of those headers, an automatic header test to check for self
containment will be generated. Private headers should be placed under the
``src`` directory. This allows for clear seperation. The ``cmake`` subdirectory
may include additional |cmake|_ scripts needed to generate the respective build
configurations.

Documentation is placed in the ``docs`` folder. A empty skeleton for the index
is created, which is picked up by the main build system and will be part of the
generated documentation. Each header inside the ``include`` directory will
automatically be processed by Doxygen and included into the documentation. If a
header should be excluded from the API reference, a comment ``//
sphinx:undocumented`` needs to be added.

In order to consume any library defined here, all you have to do is use
``target_link_libraries`` to get the dependencies. This of course requires that
the library to link against specified the appropriate target include directories
and libraries.
