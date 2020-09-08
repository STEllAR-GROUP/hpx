..
    Copyright (c) 2019 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _modules_execution_base:

===============
execution_base
===============

The basic execution module is the main entry point to implement parallel and
concurrent operations. It is modeled after `P0443 <http://wg21.link/p0443>`_
with some additions and implementations for the described concepts. Most
notably, it provides an abstraction for execution resources, execution contexts
and execution agents in such a way, that it provides customization points that
those aforementioned concepts can be replaced and combined with ease.

For that purpose, three virtual base classes are provided to be able to provide
implementations with different properties:

 - ``resource_base``: This is the abstraction for execution resources, that is
    for example CPU cores or an accelerator.
 - ``context_base``: An execution context uses execution resources and is able
    to spawn new execution agents, as new threads of executions on the available
    resources.
 - ``agent_base``: The execution agent represents the thread of execution, and
    can be used to yield, suspend, resume or abort a thread of execution.
