..
    Copyright (C) 2018 Mikael Simberg

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

Building documentation
======================

Please see the :ref:`documentation prerequisites <documentation_prerequisites>`
section for details on what you need in order to build the |hpx| documentation.
Enable building of the documentation by setting ``HPX_WITH_DOCUMENTATION=ON``
during |cmake|_ configuration. To build the documentation build the ``docs``
target using your build tool.

.. note::

   If you add new source files to the Sphinx documentation you have to run
   |cmake| again to have the files included in the build.


Style guide
===========

The documentation is written using reStructuredText. These are the conventions
used for formatting the documentation:

* Use at most 80 characters per line.
* Top-level headings use over- and underlines with ``=``.
* Sub-headings use only underlines with characters in decreasing level of
  importance: ``=``, ``-`` and ``.``.
* Use sentence case in headings.
* Refer to common terminology using ``:term:`Component```.
* Indent content of directives (``.. directive::``) by three spaces.
* For C++ code samples at the end of paragraphs, use ``::`` and indent the code
  sample by 4 spaces.

  * For other languages (or if you don't want a colon at the end of the
    paragraph) use ``.. code-block:: language`` and indent by three spaces as
    with other directives.
* Use ``.. list-table::`` to wrap tables with a lot of text in cells.

API documentation
=================

The source code is documented using |doxygen|_. If you add new API documentation
either to existing or new source files, make sure that you add the documented
source files to the ``doxygen_dependencies`` variable in
``docs/CMakeLists.txt``.
