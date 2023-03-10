..
    Copyright (c) 2023 Dimitra Karatza

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _building_tests_examples:

===========================
Building tests and examples
===========================

.. _tests:

Tests
=====

To build the tests:

.. code-block:: shell-session

    $ cmake --build . --target tests

To control which tests to run use ``ctest``:

* To run single tests, for example a test for ``for_loop``:

.. code-block:: shell-session

    $ ctest --output-on-failure -R tests.unit.modules.algorithms.algorithms.for_loop

* To run a whole group of tests:

.. code-block:: shell-session

    $ ctest --output-on-failure -R tests.unit

.. _examples:

Examples
========

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
