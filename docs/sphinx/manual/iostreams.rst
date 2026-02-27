..
    Copyright (C)      2017 Adrian Serio
    Copyright (C) 2007-2015 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _iostreams:

===============================
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
   ``hpx::cout`` and ``hpx::cerr``.

In order to use either ``hpx::cout`` or ``hpx::cerr``, application codes need to
``#include <hpx/include/iostreams.hpp>``. For an example, please see the
following 'Hello world' program:

.. literalinclude:: ../../examples/quickstart/hello_world_1.cpp
   :language: c++
   :start-after: //[hello_world_1_getting_started
   :end-before: //]

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
