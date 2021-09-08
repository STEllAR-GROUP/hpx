..
    Copyright (C) 2018 Mikael Simberg

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _documentation:

=============
Documentation
=============

This documentation is built using |sphinx|_, and an automatically generated API
reference using |doxygen|_ and |breathe|_.

We always welcome suggestions on how to improve our documentation, as well as
pull requests with corrections and additions.

Prerequisites
=============

To build the |hpx| documentation, you need recent versions of the following
packages:

- ``python3``
- ``sphinx 3.5.4`` (Python package)
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


Building documentation
======================

Enable building of the documentation by setting ``HPX_WITH_DOCUMENTATION=ON``
during |cmake|_ configuration. To build the documentation, build the ``docs``
target using your build tool. The default output format is HTML documentation.
You can choose alternative output formats (single-page HTML, PDF, and man) with
the ``HPX_WITH_DOCUMENTATION_OUTPUT_FORMATS`` CMake option.

.. note::

   If you add new source files to the Sphinx documentation, you have to run
   CMake again to have the files included in the build.


Style guide
===========

The documentation is written using reStructuredText. These are the conventions
used for formatting the documentation:

* Use, at most, 80 characters per line.
* Top-level headings use over- and underlines with ``=``.
* Sub-headings use only underlines with characters in decreasing level of
  importance: ``=``, ``-`` and ``.``.
* Use sentence case in headings.
* Refer to common terminology using ``:term:`Component```.
* Indent content of directives (``.. directive::``) by three spaces.
* For C++ code samples at the end of paragraphs, use ``::`` and indent the code
  sample by 4 spaces.

  * For other languages (or if you don't want a colon at the end of the
    paragraph), use ``.. code-block:: language`` and indent by three spaces as
    with other directives.
* Use ``.. list-table::`` to wrap tables with a lot of text in cells.

API documentation
=================

The source code is documented using Doxygen. If you add new API documentation
either to existing or new source files, make sure that you add the documented
source files to the ``doxygen_dependencies`` variable in
``docs/CMakeLists.txt``.
