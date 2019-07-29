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
can be used to generate a basic skeleton. To create a library skeleton, run the
tool in the ``libs`` subdirectory with the module name as an argument:

.. code-block:: bash

    ./create_library_skeleton <lib_name>

This creates a skeleton with the necessary files for an |hpx| module. It will not create any actual source files. The structure of this skeleton is as follows:

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

A main ``CMakeLists.txt`` is created in the root directory of the module. By
default it contains a call to ``add_hpx_module`` which takes care of most of the
boilerplate required for a module. You only need to fill in the source and
header files in most cases.

``add_hpx_module`` requires a module name. Optional flags are:

* ``DEPRECATION_WARNINGS``: Enables deprecation warnings for the module.

Optional single-value arguments are:

* ``COMPATIBILITY_HEADERS``: Can be ``ON``, ``OFF``, or left out. Enables
  compatibility headers. Creates a variable which can be turned on or off by the
  user when set to ``ON`` or ``OFF``. If left out the option is completely
  disabled.
* ``INSTALL_BINARIES``: Install the resulting library.

Optional multi-value arguments-are:

* ``SOURCES``: List of source files.
* ``HEADERS``: List of header files.
* ``COMPAT_HEADERS``: List of compatibility header files.
* ``DEPENDENCIES``: Libraries that this module depends on, such as other modules.
* ``CMAKE_SUBDIRS``: List of subdirectories to add to the module.

The ``include`` directory should contain only headers that other libraries need.
For each of those headers, an automatic header test to check for self
containment will be generated. Private headers should be placed under the
``src`` directory. This allows for clear seperation. The ``cmake`` subdirectory
may include additional |cmake|_ scripts needed to generate the respective build
configurations.

Compatibility headers (forwarding headers for headers whose location is changed
when creating a module, if moving them from the main library) should be placed
in an ``include_compatibility`` directory. This directory is not created by
default.

Documentation is placed in the ``docs`` folder. A empty skeleton for the index
is created, which is picked up by the main build system and will be part of the
generated documentation. Each header inside the ``include`` directory will
automatically be processed by Doxygen and included into the documentation. If a
header should be excluded from the API reference, a comment ``//
sphinx:undocumented`` needs to be added.

Tests are placed in suitable subdirectories of ``tests``.

When in doubt, consult existing modules for examples on how to structure the
module.
