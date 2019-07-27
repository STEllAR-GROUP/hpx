..
    Copyright (C) 2019 Tapasweni Pathak

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _hpx_functions:

=============
|hpx| functions
=============

This document describes the prototypes, the allowable arguments of the listed
|hpx| functions.

async
=====

``async`` in C++ allows running the function f asynchronously and returns a
``hpx::future`` having the result of that function call.

``async`` calls a function f with arguments args as per launch policy. You can
have async flag set or deferred flag set.

Implementing async functions should have input parameters as

- Function/Function Object/Lambda
- Executor function
- Launch policy

return value is ``hpx::future``.

.. literalinclude:: ../../examples/async.cpp

You can chain multiple async operations. For implementing chain async programming
``objFuture.then(...).then(...).then(...).then(...).then(...)``.

Throws ``hpx::system_error`` with error condition ``hpx::errc::resource_unavailable_try_again`` if the launch policy equals ``hpx::launch::async`` and the implementation is unable to start a new thread (if the policy is async|deferred or has additional bits set, it will fall back to deferred or the implementation-defined policies in this case), or ``hpx::bad_alloc`` if memory for the internal data structures could not be allocated.

then
====

then in asychronous programming is used in chained async implementations. Using
then function will avoid blocking waits or wasting threads on polling. Future
can have then chained implementations. With then the antecedent future is ready
(has a value or exception stored) before the continuation starts as instructed
by lambda function. Example implementation using hpx is

.. literalinclude:: ../../examples/then.cpp

If implicit unwrapping takes place and the continuation returns an invalid future, then the shared state is made ready with an exception of type ``hpx::future_error`` with an error condition of ``hpx::future_errc::broken_promise``.

wait
====

``hpx::future<T>::wait`` blocks the function implementation until the call is
available. This function call can be checked using ``valid() == true``. Example
implementation using hpx is

.. literalinclude:: ../../examples/wait.cpp

wait_all
========

The function ``wait_all`` is an operator allowing to join on the result of all
given futures. It AND-composes all future objects given and returns after they
finished executing. The function ``wait_all`` returns after all futures have
become ready. All input futures are still valid after wait all returns.

.. literalinclude:: ../../hpx/lcos/wait_all.hpp
   :lines: 48-91

.. literalinclude:: ../../examples/wait_all.cpp
