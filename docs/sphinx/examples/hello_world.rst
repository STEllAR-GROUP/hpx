..
    Copyright (C) 2012 Adrian Serio
    Copyright (C) 2012 Vinay C Amatya
    Copyright (C) 2015 Hartmut Kaiser

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _examples_hello_world:

==========================================
Remote execution with actions: Hello world
==========================================

This program will print out a hello world message on every OS-thread on every
:term:`locality`. The output will look something like this:

.. code-block:: text

   hello world from OS-thread 1 on locality 0
   hello world from OS-thread 1 on locality 1
   hello world from OS-thread 0 on locality 0
   hello world from OS-thread 0 on locality 1

Setup
=====

The source code for this example can be found here: :download:`hello_world.cpp
<../../examples/quickstart/hello_world.cpp>`.

To compile this program, go to your |hpx| build directory (see
:ref:`hpx_build_system` for information on configuring and building |hpx|) and
enter:

.. code-block:: bash

   make examples.quickstart.hello_world

To run the program type:

.. code-block:: bash

   ./bin/hello_world

This should print:

.. code-block:: text

   hello world from OS-thread 0 on locality 0

To use more OS-threads use the command line option :option:`--hpx:threads` and
type the number of threads that you wish to use. For example, typing:

.. code-block:: bash

   ./bin/hello_world --hpx:threads 2

will yield:

.. code-block:: text

   hello world from OS-thread 1 on locality 0
   hello world from OS-thread 0 on locality 0

Notice how the ordering of the two print statements will change with
subsequent runs. To run this program on multiple localities please see the
section :ref:`unix_pbs`.

Walkthrough
===========

Now that you have compiled and run the code, let's look at how the code works,
beginning with ``main()``:

.. literalinclude:: ../../examples/quickstart/hello_world.cpp
   :lines: 130-156

In this excerpt of the code we again see the use of futures. This time the
futures are stored in a vector so that they can easily be accessed.
:cpp:func:`hpx::wait_all` is a family of functions that wait on for an
``std::vector<>`` of futures to become ready. In this piece of code, we are
using the synchronous version of :cpp:func:`hpx::wait_all()`, which takes one
argument (the ``std::vector<>`` of futures to wait on). This function will not
return until all the futures in the vector have been executed.

In :ref:`examples_fibonacci` we used :cpp:func:`hpx::find_here()` to specify the
target of our actions. Here, we instead use
:cpp:func:`hpx::find_all_localities()`, which returns an ``std::vector<>``
containing the identifiers of all the machines in the system, including the one
that we are on.

As in :ref:`examples_fibonacci` our futures are set using
:cpp:func:`hpx::async\<>()`. The ``hello_world_foreman_action`` is declared
here:

.. literalinclude:: ../../examples/quickstart/hello_world.cpp
   :lines: 123-125

Another way of thinking about this wrapping technique is as follows: functions
(the work to be done) are wrapped in actions, and actions can be executed
locally or remotely (e.g. on another machine participating in the computation).

Now it is time to look at the ``hello_world_foreman()`` function which was
wrapped in the action above:

.. literalinclude:: ../../examples/quickstart/hello_world.cpp
   :lines: 66-119

Now, before we discuss ``hello_world_foreman()``, let's talk about the
:cpp:func:`hpx::wait_each()` function. :cpp:func:`hpx::lcos::wait_each` for each
one. The version of :cpp:func:`hpx::lcos::wait_each` invokes a callback function
provided by the user, supplying the callback function with the result of the
future.

In ``hello_world_foreman()``, an ``std::set<>`` called ``attendance`` keeps
track of which OS-threads have printed out the hello world message. When the
OS-thread prints out the statement, the future is marked as ready, and
:cpp:func:`hpx::lcos::wait_each` in ``hello_world_foreman()``. If it is not
executing on the correct OS-thread, it returns a value of -1, which causes
``hello_world_foreman()`` to leave the OS-thread id in ``attendance``.

.. literalinclude:: ../../examples/quickstart/hello_world.cpp
   :lines: 37-61

Because |hpx| features work stealing task schedulers, there is no way to
guarantee that an action will be scheduled on a particular OS-thread. This is
why we must use a guess-and-check approach.
