..
    Copyright (C) 2007-2013 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _terminology:

===========
Terminology
===========

This section gives definitions for some of the terms used throughout the
|hpx| documentation and source code.

.. glossary::

   Locality

      A locality in |hpx| describes a synchronous domain of execution, or the
      domain of bounded upper response time. This normally is just a single node
      in a cluster or a NUMA domain in a SMP machine.

   Active Global Address Space
   AGAS

      |hpx| incorporates a global address space. Any executing thread can access
      any object within the domain of the parallel application with the caveat
      that it must have appropriate access privileges. The model does not assume
      that global addresses are cache coherent; all loads and stores will deal
      directly with the site of the target object. All global addresses within a
      Synchronous Domain are assumed to be cache coherent for those processor
      cores that incorporate transparent caches. The Active Global Address Space
      used by |hpx| differs from research |pgas|_ models. Partitioned Global
      Address Space is passive in their means of address translation. Copy
      semantics, distributed compound operations, and affinity relationships are
      some of the global functionality supported by AGAS.

   Process

      The concept of the "process" in |hpx| is extended beyond that of either
      sequential execution or communicating sequential processes. While the
      notion of process suggests action (as do "function" or "subroutine") it
      has a further responsibility of context, that is, the logical container of
      program state. It is this aspect of operation that process is employed in
      |hpx|. Furthermore, referring to "parallel processes" in |hpx| designates
      the presence of parallelism within the context of a given process, as well
      as the coarse grained parallelism achieved through concurrency of multiple
      processes of an executing user job. |hpx| processes provide a hierarchical
      name space within the framework of the active global address space and
      support multiple means of internal state access from external sources.

   Parcel

      The Parcel is a component in |hpx| that communicates data, invokes an
      action at a distance, and distributes flow-control through the migration
      of continuations. Parcels bridge the gap of asynchrony between synchronous
      domains while maintaining symmetry of semantics between local and global
      execution. Parcels enable message-driven computation and may be seen as a
      form of "active messages". Other important forms of message-driven
      computation predating active messages include `dataflow tokens
      <http://en.wikipedia.org/wiki/Dataflow_architecture>`_, the `J-machine's
      <http://en.wikipedia.org/wiki/J%E2%80%93Machine>`_ support for remote
      method instantiation, and at the coarse grained variations of Unix remote
      procedure calls, among others. This enables work to be moved to the data
      as well as performing the more common action of bringing data to the work.
      A parcel can cause actions to occur remotely and asynchronously, among
      which are the creation of threads at different system nodes or synchronous
      domains.

   Local Control Object
   Lightweight Control Object
   LCO

      A local control object (sometimes called a lightweight control object) is
      a general term for the synchronization mechanisms used in |hpx|. Any
      object implementing a certain concept can be seen as an LCO. This concepts
      encapsulates the ability to be triggered by one or more events which when
      taking the object into a predefined state will cause a thread to be
      executed. This could either create a new thread or resume an existing
      thread.

      The LCO is a family of synchronization functions potentially representing
      many classes of synchronization constructs, each with many possible
      variations and multiple instances. The LCO is sufficiently general that it
      can subsume the functionality of conventional synchronization primitives
      such as spinlocks, mutexes, semaphores, and global barriers. However due
      to the rich concept an LCO can represent powerful synchronization and
      control functionality not widely employed, such as dataflow and futures
      (among others), which open up enormous opportunities for rich diversity of
      distributed control and operation.

      See :ref:`lcos` for more details on how to use LCOs in |hpx|.

   Action

      An action is a function that can be invoked remotely. In |hpx| a plain
      function can be made into an action using a macro. See
      :ref:`applying_actions` for details on how to use actions in |hpx|.

   Component

      A component is a C++ object which can be accessed remotely. A component
      can also contain member functions which can be invoked remotely. These are
      referred to as component actions. See :ref:`components` for details on how
      to use components in |hpx|.
