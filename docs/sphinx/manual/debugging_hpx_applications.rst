..
    Copyright (C) 2018 Mikael Simberg

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

============================
Debugging |hpx| applications
============================

Using a debugger with |hpx| applications
========================================

Using a debugger such as ``gdb`` with |hpx| applications is no problem. However,
there are some things to keep in mind to make the experience somewhat more
productive.

Call stacks in |hpx| can often be quite unwieldy as the library is heavily
templated and the call stacks can be very deep. For this reason it is sometimes
a good idea compile |hpx| in ``RelWithDebInfo`` mode, which applies some
optimizations but keeps debugging symbols. This can often compress call stacks
significantly. On the other hand, stepping through the code can also be more
difficult because of statements being reordered and variables being optimized
away. Also, note that because |hpx| implements user-space threads and context
switching, call stacks may not always be complete in a debugger.

|hpx| launches not only worker threads but also a few helper threads. The first
thread is the main thread, which typically does no work in an |hpx| application,
except at startup and shutdown. If using the default settings, |hpx| will spawn
six additional threads (used for service thread pools). The first worker thread
is usually the eighth thread, and most user codes will be run on these worker
threads. The last thread is a helper thread used for |hpx| shutdown.

Finally, since |hpx| is a multi-threaded runtime, the following ``gdb`` options
can be helpful:

.. code-block:: text

   set pagination off
   set non-stop on

Non-stop mode allows users to have a single thread stop on a breakpoint without
stopping all other threads as well.


Using sanitizers with |hpx| applications
========================================

.. warning::

   Not all parts of |hpx| are sanitizer clean. This means that users may end up
   with false positives from |hpx| itself when using sanitizers for their
   applications.

To use sanitizers with |hpx|, turn on ``HPX_WITH_SANITIZERS`` and turn
off ``HPX_WITH_STACKOVERFLOW_DETECTION`` during |cmake| configuration. It's
recommended to also build |boost| with the same sanitizers that will be
used for |hpx|. The appropriate sanitizers can then be enabled using |cmake| by
appending ``-fsanitize=address -fno-omit-frame-pointer`` to ``CMAKE_CXX_FLAGS``
and ``-fsanitize=address`` to ``CMAKE_EXE_LINKER_FLAGS``. Replace ``address``
with the sanitizer that you want to use.
