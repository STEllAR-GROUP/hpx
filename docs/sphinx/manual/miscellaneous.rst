..
    Copyright (C)      2017 Adrian Serio
    Copyright (C) 2007-2015 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

=============
Miscellaneous
=============

.. _error_handling:

Error handling
==============

Like in any other asynchronous invocation scheme, it is important to be able to
handle error conditions occurring while the asynchronous (and possibly remote)
operation is executed. In |hpx| all error handling is based on standard C++
exception handling. Any exception thrown during the execution of an asynchronous
operation will be transferred back to the original invocation :term:`locality`,
where it will be rethrown during synchronization with the calling thread.

The source code for this example can be found here:
:download:`error_handling.cpp <../../examples/quickstart/error_handling.cpp>`.

.. _exceptions:

Working with exceptions
-----------------------

For the following description assume that the function ``raise_exception()``
is executed by invoking the plain action ``raise_exception_type``.

.. literalinclude:: ../../examples/quickstart/error_handling.cpp
   :language: c++
   :lines: 13-17

The exception is thrown using the macro :c:macro:`HPX_THROW_EXCEPTION`. The type
of the thrown exception is :cpp:class:`hpx::exception`. This associates
additional diagnostic information with the exception, such as file name and line
number, :term:`locality` id and thread id, and stack backtrace from the point
where the exception was thrown.

Any exception thrown during the execution of an action is transferred back to
the (asynchronous) invocation site. It will be rethrown in this context when the
calling thread tries to wait for the result of the action by invoking either
``future<>::get()`` or the synchronous action invocation wrapper as shown here:

.. literalinclude:: ../../examples/quickstart/error_handling.cpp
   :language: c++
   :lines: 27-42

.. note::

   The exception is transferred back to the invocation site even if it is
   executed on a different :term:`locality`.

Additionally, this example demonstrates how an exception thrown by an (possibly
remote) action can be handled. It shows the use of
:cpp:func:`hpx::diagnostic_information`, which retrieves all available diagnostic
information from the exception as a formatted string. This includes, for
instance, the name of the source file and line number, the sequence number of
the OS thread and the |hpx| thread id, the :term:`locality` id and the stack
backtrace of the point where the original exception was thrown.

Under certain circumstances it is desirable to output only some of the
diagnostics, or to output those using different formatting. For this case, |hpx|
exposes a set of lower-level functions as demonstrated in the following code
snippet:

.. literalinclude:: ../../examples/quickstart/error_handling.cpp
   :language: c++
   :lines: 47-72

.. _error_code:

Working with error codes
------------------------

Most of the API functions exposed by |hpx| can be invoked in two different
modes. By default those will throw an exception on error as described above.
However, sometimes it is desirable not to throw an exception in case of an error
condition. In this case an object instance of the :cpp:class:`hpx::error_code`
type can be passed as the last argument to the API function. In case of an error,
the error condition will be returned in that :cpp:class:`hpx::error_code`
instance. The following example demonstrates extracting the full diagnostic
information without exception handling:

.. literalinclude:: ../../examples/quickstart/error_handling.cpp
   :language: c++
   :lines: 79-100

.. note::

   The error information is transferred back to the invocation site even if it
   is executed on a different :term:`locality`.

This example show how an error can be handled without having to resolve to
exceptions and that the returned :cpp:class:`hpx::error_code` instance can be
used in a very similar way as the :cpp:class:`hpx::exception` type above. Simply
pass it to the :cpp:func:`hpx::diagnostic_information`, which retrieves all
available diagnostic information from the error code instance as a formatted
string.

As for handling exceptions, when working with error codes, under certain
circumstances it is desirable to output only some of the diagnostics, or to
output those using different formatting. For this case, |hpx| exposes a set of
lower-level functions usable with error codes as demonstrated in the following
code snippet:

.. literalinclude:: ../../examples/quickstart/error_handling.cpp
   :language: c++
   :lines: 107-139

For more information please refer to the documentation of
:cpp:func:`hpx::get_error_what`, :cpp:func:`hpx::get_error_locality_id`,
:cpp:func:`hpx::get_error_host_name`, :cpp:func:`hpx::get_error_process_id`,
:cpp:func:`hpx::get_error_function_name`, :cpp:func:`hpx::get_error_file_name`,
:cpp:func:`hpx::get_error_line_number`, :cpp:func:`hpx::get_error_os_thread`,
:cpp:func:`hpx::get_error_thread_id`,
:cpp:func:`hpx::get_error_thread_description`,
:cpp:func:`hpx::get_error_backtrace`, :cpp:func:`hpx::get_error_env`, and
:cpp:func:`hpx::get_error_state`.

.. _lightweight_error_code:

Lightweight error codes
-----------------------

Sometimes it is not desirable to collect all the ambient information about the
error at the point where it happened as this might impose too much overhead for
simple scenarios. In this case, |hpx| provides a lightweight error code facility
that will hold the error code only. The following snippet demonstrates its use:

.. literalinclude:: ../../examples/quickstart/error_handling.cpp
   :language: c++
   :lines: 146-166

All functions that retrieve other diagnostic elements from the
:cpp:class:`hpx::error_code` will fail if called with a lightweight error_code
instance.

.. _utilities:

Utilities in |hpx|
==================

In order to ease the burden of programming, |hpx| provides several
utilities to users. The following section documents those facilies.

.. _checkpoint:

Checkpoint
----------

See :ref:`modules_checkpoint`.

.. _iostreams:

The |hpx| I/O-streams component
===============================

The |hpx| I/O-streams subsystem extends the standard C++ output streams
``std::cout`` and ``std::cerr`` to work in the distributed setting of an |hpx|
application. All of the output streamed to ``hpx::cout`` will be dispatched to
``std::cout`` on the console :term:`locality`. Likewise, all output generated
from ``hpx::cerr`` will be dispatched to ``std::cerr`` on the console
:term:`locality`.

.. note::

   All existing standard manipulators can be used in conjunction with
   ``hpx::cout`` and ``hpx::cerr`` Historically, |hpx| also defines
   ``hpx::endl`` and ``hpx::flush`` but those are just aliases for the
   corresponding standard manipulators.

In order to use either ``hpx::cout`` or ``hpx::cerr``, application codes need to
``#include <hpx/include/iostreams.hpp>``. For an example, please see the
following 'Hello world' program:

.. literalinclude:: ../../examples/quickstart/hello_world_1.cpp
   :language: c++

Additionally, those applications need to link with the iostreams component. When
using CMake this can be achieved by using the ``COMPONENT_DEPENDENCIES``
parameter; for instance:

.. code-block:: cmake

   include(HPX_AddExecutable)

   add_hpx_executable(
       hello_world
       SOURCES hello_world.cpp
       COMPONENT_DEPENDENCIES iostreams
   )

.. note::

   The ``hpx::cout`` and ``hpx::cerr`` streams buffer all output locally until a
   ``std::endl`` or ``std::flush`` is encountered. That means that no output
   will appear on the console as long as either of these is explicitly used.
