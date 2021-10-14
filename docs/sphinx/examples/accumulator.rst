..
    Copyright (C) 2012 Adrian Serio
    Copyright (C) 2012 Vinay C Amatya
    Copyright (C) 2015 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _examples_accumulator:

======================
Components and actions
======================

The accumulator example demonstrates the use of components. Components are C++
classes that expose methods as a type of |hpx| action. These actions are called
component actions.

Components are globally named, meaning that a component action can be called
remotely (e.g.,  from another machine). There are two accumulator examples in
|hpx|.

In the :ref:`examples_fibonacci` and the :ref:`examples_hello_world`, we
introduced plain actions, which wrapped global functions. The target of a plain
action is an identifier which refers to a particular machine involved in the
computation. For plain actions, the target is the machine where the action will
be executed.

Component actions, however, do not target machines. Instead, they target
component instances. The instance may live on the machine that we've invoked the
component action from, or it may live on another machine.

The component in this example exposes three different functions:

* ``reset()`` - Resets the accumulator value to 0.
* ``add(arg)`` - Adds ``arg`` to the accumulators value.
* ``query()`` - Queries the value of the accumulator.

This example creates an instance of the accumulator, and then allows the user to
enter commands at a prompt, which subsequently invoke actions on the accumulator
instance.

Setup
=====

The source code for this example can be found here:
:download:`accumulator_client.cpp
<../../examples/accumulators/accumulator_client.cpp>`.

To compile this program, go to your |hpx| build directory (see
:ref:`hpx_build_system` for information on configuring and building |hpx|) and
enter:

.. code-block:: shell-session

   $ make examples.accumulators.accumulator

To run the program type:

.. code-block:: shell-session

   $ ./bin/accumulator_client

Once the program starts running, it will print the following prompt and then
wait for input. An example session is given below:

.. code-block:: text

   commands: reset, add [amount], query, help, quit
   > add 5
   > add 10
   > query
   15
   > add 2
   > query
   17
   > reset
   > add 1
   > query
   1
   > quit

Walkthrough
===========

Now, let's take a look at the source code of the accumulator example. This
example consists of two parts: an |hpx| component library (a library that
exposes an |hpx| component) and a client application which uses the library.
This walkthrough will cover the |hpx| component library. The code for the client
application can be found here: :download:`accumulator_client.cpp
<../../examples/accumulators/accumulator_client.cpp>`.

An |hpx| component is represented by two C++ classes:

* **A server class** - The implementation of the component's functionality.
* **A client class** - A high-level interface that acts as a proxy for an
  instance of the component.

Typically, these two classes both have the same name, but the server class
usually lives in different sub-namespaces (``server``). For example, the full
names of the two classes in accumulator are:

* ``examples::server::accumulator`` (server class)
* ``examples::accumulator`` (client class)

The server class
----------------

The following code is from: :download:`accumulator.hpp
<../../examples/accumulators/server/accumulator.hpp>`.

All |hpx| component server classes must inherit publicly from the |hpx|
component base class: :cpp:class:`hpx::components::component_base`

The accumulator component inherits from
:cpp:class:`hpx::components::locking_hook`. This allows the runtime system to
ensure that all action invocations are serialized. That means that the system
ensures that no two actions are invoked at the same time on a given component
instance. This makes the component thread safe and no additional locking has to
be implemented by the user. Moreover, an accumulator component is a component
because it also inherits from :cpp:class:`hpx::components::component_base` (the
template argument passed to locking_hook is used as its base class). The
following snippet shows the corresponding code:

.. literalinclude:: ../../examples/accumulators/server/accumulator.hpp
   :language: c++
   :start-after: //[accumulator_server_inherit
   :end-before: //]

Our accumulator class will need a data member to store its value in, so let's
declare a data member:

.. literalinclude:: ../../examples/accumulators/server/accumulator.hpp
   :language: c++
   :start-after: //[accumulator_server_data_member
   :end-before: //]

The constructor for this class simply initializes ``value_`` to 0:

.. literalinclude:: ../../examples/accumulators/server/accumulator.hpp
   :language: c++
   :start-after: //[accumulator_server_ctor
   :end-before: //]

Next, let's look at the three methods of this component that we will be exposing
as component actions:

.. literalinclude:: ../../examples/accumulators/server/accumulator.hpp
   :language: c++
   :start-after: //[accumulator_components
   :end-before: //]

Here are the action types. These types wrap the methods we're exposing. The
wrapping technique is very similar to the one used in the
:ref:`examples_fibonacci` and the :ref:`examples_hello_world`:

.. literalinclude:: ../../examples/accumulators/server/accumulator.hpp
   :language: c++
   :start-after: //[accumulator_action_types
   :end-before: //]

The last piece of code in the server class header is the declaration of the
action type registration code:

.. literalinclude:: ../../examples/accumulators/server/accumulator.hpp
   :language: c++
   :start-after: //[accumulator_registration_declarations
   :end-before: //]

.. note::

   The code above must be placed in the global namespace.

The rest of the registration code is in
:download:`accumulator.cpp <../../examples/accumulators/accumulator.cpp>`

.. literalinclude:: ../../examples/accumulators/accumulator.cpp
   :language: c++
   :start-after: //[accumulator_registration_definitions
   :end-before: //]


.. note::

   The code above must be placed in the global namespace.

The client class
----------------

The following code is from :download:`accumulator.hpp
<../../examples/accumulators/accumulator.hpp>`.

The client class is the primary interface to a component instance. Client classes
are used to create components::

    // Create a component on this locality.
    examples::accumulator c = hpx::new_<examples::accumulator>(hpx::find_here());

and to invoke component actions::

    c.add(hpx::launch::apply, 4);

Clients, like servers, need to inherit from a base class, this time,
:cpp:class:`hpx::components::client_base`:

.. literalinclude:: ../../examples/accumulators/accumulator.hpp
   :language: c++
   :start-after: //[accumulator_client_inherit
   :end-before: //]

For readability, we typedef the base class like so:

.. literalinclude:: ../../examples/accumulators/accumulator.hpp
   :language: c++
   :start-after: //[accumulator_base_type
   :end-before: //]

Here are examples of how to expose actions through a client class:

There are a few different ways of invoking actions:

* **Non-blocking**: For actions that don't have return types, or when we do not
  care about the result of an action, we can invoke the action using
  fire-and-forget semantics. This means that once we have asked |hpx| to compute
  the action, we forget about it completely and continue with our computation.
  We use :cpp:func:`hpx::apply` to invoke an action in a non-blocking fashion.

.. literalinclude:: ../../examples/accumulators/accumulator.hpp
   :language: c++
   :start-after: //[accumulator_client_reset_non_blocking
   :end-before: //]

* **Asynchronous**: Futures, as demonstrated in :ref:`examples_fibonacci_local`,
  :ref:`examples_fibonacci`, and the :ref:`examples_hello_world`, enable
  asynchronous action invocation. Here's an example from the accumulator client
  class:

.. literalinclude:: ../../examples/accumulators/accumulator.hpp
   :language: c++
   :start-after: //[accumulator_client_query_async
   :end-before: //]

* **Synchronous**: To invoke an action in a fully synchronous manner, we can
  simply call :cpp:func:`hpx::async`\ ``().get()`` (i.e., create a future and
  immediately wait on it to be ready). Here's an example from the accumulator
  client class:

.. literalinclude:: ../../examples/accumulators/accumulator.hpp
   :language: c++
   :start-after: //[accumulator_client_add_sync
   :end-before: //]

Note that ``this->get_id()`` references a data member of the
:cpp:class:`hpx::components::client_base` base class which identifies the server
accumulator instance.

:cpp:class:`hpx::naming::id_type` is a type which represents a global identifier
in |hpx|. This type specifies the target of an action. This is the type that is
returned by :cpp:func:`hpx::find_here` in which case it represents the
:term:`locality` the code is running on.
