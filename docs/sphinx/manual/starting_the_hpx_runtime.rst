..
    Copyright (C) 2018 Nikunj Gupta
    Copyright (C) 2007-2017 Hartmut Kaiser

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _starting_hpx:

==========================
Starting the |hpx| runtime
==========================

In order to write an application which uses services from the |hpx| runtime
system you need to initialize the |hpx| library by inserting certain calls
into the code of your application. Depending on your use case, this can be done
in 3 different ways:

* :ref:`Minimally invasive <minimal>`: Re-use the ``main()`` function as the
  main |hpx| entry point.
* :ref:`Balanced use case <medium>`: Supply your own main |hpx| entry point
  while blocking the main thread.
* :ref:`Most flexibility <flexible>`: Supply your own main |hpx| entry point
  while avoiding to block the main thread.
* :ref:`Suspend and resume <suspend_resume>`: As above but suspend and resume
  the |hpx| runtime to allow for other runtimes to be used.

.. _minimal:

Re-use the ``main()`` function as the main |hpx| entry point
============================================================

This method is the least intrusive to your code. It however provides you with
the smallest flexibility in terms of initializing the |hpx| runtime system. The
following code snippet shows what a minimal |hpx| application using this
technique looks like:

.. code-block:: c++

    #include <hpx/hpx_main.hpp>

    int main(int argc, char* argv[])
    {
        return 0;
    }

The only change to your code you have to make is to include the file
``hpx/hpx_main.hpp``. In this case the function ``main()`` will be invoked as
the first |hpx| thread of the application. The runtime system will be
initialized behind the scenes before the function ``main()`` is executed and
will automatically stop after ``main()`` has returned. All |hpx| API functions
can be used from within this function now.

.. note::

   The function ``main()`` does not need to expect receiving ``argc`` ``argv``
   as shown above, but could expose the signature ``int main()``. This is
   consistent with the usually allowed prototypes for the function ``main()`` in
   C++ applications.

All command line arguments specific to |hpx| will still be processed by the
|hpx| runtime system as usual. However, those command line options will be
removed from the list of values passed to ``argc``/\ ``argv`` of the function
``main()``. The list of values passed to ``main()`` will hold only the
commandline options which are not recognized by the |hpx| runtime system (see
the section :ref:`commandline` for more details on what options are recognized
by |hpx|).

.. note::

   In this mode all one-letter-shortcuts are disabled which are normally
   available on the |hpx| command line (such as ``-t`` or ``-l`` see
   :ref:`commandline`). This is done to minimize any possible interaction
   between the command line options recognized by the |hpx| runtime system and
   any command line options defined by the application.

The value returned from the function ``main()`` as shown above will be returned
to the operating system as usual.

.. important::

   To achieve this seamless integration, the header file ``hpx/hpx_main.hpp``
   defines a macro::

        #define main hpx_startup::user_main

   which could result in unexpected behavior.

.. important::

   To achieve this seamless integration, we use different implementations for
   different Operating Systems. In case of Linux or Mac OSX, the code present in
   ``hpx_wrap.cpp`` is put into action. We hook into the system function in case
   of Linux and provide alternate entry point in case of Mac OSX. For other
   Operating Systems we rely on a macro::

       #define main hpx_startup::user_main

   provided in the header file ``hpx/hpx_main.hpp``. This implementation can
   result in unexpected behavior.

.. caution::

   We make use of an *override* variable ``include_libhpx_wrap`` in the header
   file ``hpx/hpx_main.hpp`` to swiftly choose the function call stack at
   runtime. Therefore, the header file should *only* be included in the main
   executable. Including it in the components will result in multiple definition
   of the variable.

.. _medium:

Supply your own main |hpx| entry point while blocking the main thread
=====================================================================

With this method you need to provide an explicit main thread function named
``hpx_main`` at global scope. This function will be invoked as the main entry
point of your |hpx| application on the console :term:`locality` only (this
function will be invoked as the first |hpx| thread of your application). All
|hpx| API functions can be used from within this function.

The thread executing the function :cpp:func:`hpx::init` will block waiting for
the runtime system to exit. The value returned from ``hpx_main`` will be
returned from :cpp:func:`hpx::init` after the runtime system has stopped.

The function :cpp:func:`hpx::finalize` has to be called on one of the |hpx|
localities in order to signal that all work has been scheduled and the runtime
system should be stopped after the scheduled work has been executed.

This method of invoking |hpx| has the advantage of you being able to decide
which version of :cpp:func:`hpx::init` to call. This allows to pass
additional configuration parameters while initializing the |hpx| runtime system.

.. code-block:: c++

   #include <hpx/hpx_init.hpp>

   int hpx_main(int argc, char* argv[])
   {
       // Any HPX application logic goes here...
       return hpx::finalize();
   }

   int main(int argc, char* argv[])
   {
       // Initialize HPX, run hpx_main as the first HPX thread, and
       // wait for hpx::finalize being called.
       return hpx::init(argc, argv);
   }

.. note::

   The function ``hpx_main`` does not need to expect receiving ``argc``/``argv``
   as shown above, but could expose one of the following signatures::

       int hpx_main();
       int hpx_main(int argc, char* argv[]);
       int hpx_main(boost::program_options::variables_map& vm);

   This is consistent with (and extends) the usually allowed prototypes for the
   function ``main()`` in C++ applications.

The header file to include for this method of using |hpx| is
``hpx/hpx_init.hpp``.

There are many additional overloads of :cpp:func:`hpx::init` available, such as
for instance to provide your own entry point function instead of ``hpx_main``.
Please refer to the function documentation for more details (see: ``hpx/hpx_init.hpp``).

.. _flexible:

Supply your own main |hpx| entry point while avoiding to block the main thread
==============================================================================

With this method you need to provide an explicit main thread function named
``hpx_main`` at global scope. This function will be invoked as the main entry
point of your |hpx| application on the console :term:`locality` only (this
function will be invoked as the first |hpx| thread of your application). All
|hpx| API functions can be used from within this function.

The thread executing the function :cpp:func:`hpx::start` will *not* block
waiting for the runtime system to exit, but will return immediately.

.. important::

   You cannot use any of the |hpx| API functions other that
   :cpp:func:`hpx::stop` from inside your ``main()`` function.

The function :cpp:func:`hpx::finalize` has to be called on one of the |hpx|
localities in order to signal that all work has been scheduled and the runtime
system should be stopped after the scheduled work has been executed.

This method of invoking |hpx| is useful for applications where the main thread
is used for special operations, such a GUIs. The function :cpp:func:`hpx::stop`
can be used to wait for the |hpx| runtime system to exit and should be at least
used as the last function called in ``main()``. The value returned from
``hpx_main`` will be returned from :cpp:func:`hpx::stop` after the runtime
system has stopped.

.. code-block:: c++

    #include <hpx/hpx_start.hpp>

    int hpx_main(int argc, char* argv[])
    {
        // Any HPX application logic goes here...
        return hpx::finalize();
    }

    int main(int argc, char* argv[])
    {
        // Initialize HPX, run hpx_main.
        hpx::start(argc, argv);

        // ...Execute other code here...

        // Wait for hpx::finalize being called.
        return hpx::stop();
    }

.. note::

   The function ``hpx_main`` does not need to expect receiving ``argc``/``argv``
   as shown above, but could expose one of the following signatures::

       int hpx_main();
       int hpx_main(int argc, char* argv[]);
       int hpx_main(boost::program_options::variables_map& vm);

   This is consistent with (and extends) the usually allowed prototypes for the
   function ``main()`` in C++ applications.

The header file to include for this method of using |hpx| is
``hpx/hpx_start.hpp``.

There are many additional overloads of :cpp:func:`hpx::start` available, such as
for instance to provide your own entry point function instead of ``hpx_main``.
Please refer to the function documentation for more details (see:
``hpx/hpx_start.hpp``).

.. _suspend_resume:

Suspending and resuming the |hpx| runtime
=========================================

In some applications it is required to combine |hpx| with other runtimes. To
support this use case |hpx| provides two functions: :cpp:func:`hpx::suspend` and
:cpp:func:`hpx::resume`. :cpp:func:`hpx::suspend` is a blocking call which will
wait for all scheduled tasks to finish executing and then put the thread pool OS
threads to sleep. :cpp:func:`hpx::resume` simply wakes up the sleeping threads
so that they are ready to accept new work. :cpp:func:`hpx::suspend` and
:cpp:func:`hpx::resume` can be found in the header ``hpx/hpx_suspend.hpp``.

.. code-block:: c++

   #include <hpx/hpx_start.hpp>
   #include <hpx/hpx_suspend.hpp>

   int main(int argc, char* argv[])
   {

      // Initialize HPX, don't run hpx_main
       hpx::start(nullptr, argc, argv);

       // Schedule a function on the HPX runtime
       hpx::apply(&my_function, ...);

       // Wait for all tasks to finish, and suspend the HPX runtime
       hpx::suspend();

       // Execute non-HPX code here

       // Resume the HPX runtime
       hpx::resume();

       // Schedule more work on the HPX runtime

       // hpx::finalize has to be called from the HPX runtime before hpx::stop
       hpx::apply([]() { hpx::finalize(); });
       return hpx::stop();
   }

.. note::

   :cpp:func:`hpx::suspend` does not wait for :cpp:func:`hpx::finalize` to be
   called. Only call :cpp:func:`hpx::finalize` when you wish to fully stop the
   |hpx| runtime.

|hpx| also supports suspending individual thread pools and threads. For details
on how to do that see the documentation for :cpp:class:`hpx::threads::thread_pool_base`.

Automatically suspending worker threads
---------------------------------------

The previous method guarantees that the worker threads are suspended when you
ask for it and that they stay suspended. An alternative way to achieve the same
effect is to tweak how quickly |hpx| suspends its worker threads when they run
out of work. The following configuration values make sure that |hpx| idles very
quickly:

.. code-block:: ini

   hpx.max_idle_backoff_time = 1000
   hpx.max_idle_loop_count = 0

They can be set on the command line using
``--hpx:ini=hpx.max_idle_backoff_time=1000`` and
``--hpx:ini=hpx.max_idle_loop_count=0``. See :ref:`launching_and_configuring`
for more details on how to set configuration parameters.

After setting idling parameters the previous example could now be written like
this instead:

.. code-block:: c++

   #include <hpx/hpx_start.hpp>

   int main(int argc, char* argv[])
   {

      // Initialize HPX, don't run hpx_main
       hpx::start(nullptr, argc, argv);

       // Schedule some functions on the HPX runtime
       // NOTE: run_as_hpx_thread blocks until completion.
       hpx::run_as_hpx_thread(&my_function, ...);
       hpx::run_as_hpx_thread(&my_other_function, ...);

       // hpx::finalize has to be called from the HPX runtime before hpx::stop
       hpx::apply([]() { hpx::finalize(); });
       return hpx::stop();
   }

In this example each call to :cpp:func:`hpx::run_as_hpx_thread` acts as a
"parallel region".

.. _hpx_main_implementation:

Working of ``hpx_main.hpp``
===========================

In order to initialize |hpx| from ``main()``, we make use of linker tricks.

It is implemented differently for different Operating Systems. Method of
implementation is as follows:

* :ref:`Linux <hpx_main_implementation_linux>`: Using linker ``--wrap`` option.
* :ref:`Mac OSX <hpx_main_implementation_osx>`: Using the linker ``-e`` option.
* :ref:`Windows <hpx_main_implementation_windows>`: Using ``#define main
  hpx_startup::user_main``

.. _hpx_main_implementation_linux:

Linux implementation
--------------------

We make use of the Linux linker ``ld``\ 's ``--wrap`` option to wrap the
``main()`` function. This way any call to ``main()`` are redirected to our own
implementation of main. It is here that we check for the existence of
``hpx_main.hpp`` by making use of a shadow variable ``include_libhpx_wrap``. The
value of this variable determines the function stack at runtime.

The implementation can be found in ``libhpx_wrap.a``.

.. important::

   It is necessary that ``hpx_main.hpp`` be not included more than once.
   Multiple inclusions can result in multiple definition of
   ``include_libhpx_wrap``.

.. _hpx_main_implementation_osx:

Mac OSX implementation
----------------------

Here we make use of yet another linker option ``-e`` to change the entry point
to our custom entry function ``initialize_main``. We initialize the |hpx|
runtime system from this function and call main from the initialized system. We
determine the function stack at runtime by making use of the shadow variable
``include_libhpx_wrap``.

The implementation can be found in ``libhpx_wrap.a``.

.. important::

   It is necessary that ``hpx_main.hpp`` be not included more than once.
   Multiple inclusions can result in multiple definition of
   ``include_libhpx_wrap``.

.. _hpx_main_implementation_windows:

Windows implementation
----------------------

We make use of a macro ``#define main hpx_startup::user_main`` to take care of
the initializations.

This implementation could result in unexpected behaviors.
