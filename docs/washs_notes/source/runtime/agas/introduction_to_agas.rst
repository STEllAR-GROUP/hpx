=================================================
 Introduction to the Active Global Address Space
=================================================
:author: Bryce Lelbach aka "wash"
:contact: blelbach@cct.lsu.edu
:organization: LSU Center for Computation and Technology, ParalleX group
:copyright: Copyright (c) 2011

Distributed under the Boost Software License, Version 1.0. (See accompanying 
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

********
Abstract
********

One of the key elements of the ParalleX execution model is a set of constructs
which implement a distributed, global 128-bit address space accessible by all
*localities* [0]_ in a ParalleX system. These services and the address space
that they deploy are collectively known as the *Active Global Address Space
(AGAS)*.

The common currency of AGAS is the *global identifier (GID)*. The GID data
structure is a representation of a unique AGAS address. These GIDs are usually
[1]_ associated with a single *object* in the physical memory of one of the
localities participating in a ParalleX system.

One of the core services that AGAS provides is global reference counting of
GIDs. This implements garbage collection of objects registered with AGAS. When
a locality decrements a GID with a reference count of 1, AGAS will inform the
locality that it must clean up the GID. [2]_

Another key aspect of AGAS' design is the mobility of objects. Objects
registered with AGAS are intended to have the capability to move between
localities. The implementation of this particular facet of AGAS' design has
largely been unexplored at this point in time.

The administration of the AGAS address space is managed by *namespaces*. [3]_
The AGAS namespaces are a series of interconnected yet distinct registries
which *bind*, *resolve* and *unbind* key-value pairs. These namespaces form
a hierarchal *service stack*::

  +---------------+ +-------------------------------+-----------------+
  |               | |       factory namespace       |                 |
  |               | |                               |                 |
  |               | |      type id -> prefixes      |                 |
  |               | +-------------------------------+                 |
  |    symbol     | +-----------------------------+ |    component    |
  |   namespace   | |      locality namespace     | |    namespace    |
  |               | |                             | |                 |
  | string -> gid | |      locality -> prefix     | | name -> type id |
  |               | +-----------------------------+ |                 |
  |               | |      primary namespace      | |                 |
  |               | |                             | |                 |
  |               | |    gid --> local address    | |                 |
  +---------------+ +-----------------------------+ +-----------------+

In this paper, we will focus on this service stack and the GID reference
counting infrastructure. 

**********
Namespaces
**********

^^^^^^^^^^^^^^^^^
Primary Namespace
^^^^^^^^^^^^^^^^^

AGAS' most primitive construct is the *primary namespace*. The primary namespace
maps GIDs to *local addresses* [4]_. Conceptually, local addresses are tuples
consisting of a protocol-specific network address [5]_, a component type, and
64-bits of application data. Typically, this application data is a pointer
which directly or indirectly refers to a data structure in a particular worker
nodes' address space. [6]_

*********
Footnotes
*********

.. [0] A locality is a set of computing resources connected by a network that
       has a finite, bounded response time. Currently we consider uniprocessor
       and multiprocessor machines with HyperTransport (or similiar) technology
       to be a single locality. 

.. [1] In the current system, no gurantee is made that a GID will be unique among
       all AGAS namespaces, though this is generally assumed to be true. Software
       implementation of this gurantee is likely forthcoming.

.. [2] Currently, GIDS are represented by the hpx::naming::gid_type class
       Instances of this class should rarely be used directly be HPX users,
       because incrementing/decrementing their reference count involves a full
       network roundtrip to AGAS, and therefore are expensive to copy. The
       hpx::naming::id_type class is a handle for hpx::naming::gid_type which is
       reference counted locally; this is the class exposed to to user code.

.. [3] The english word 'namespace' is defined as an abstract domain that
       provides context for the elements that it holds. This definition is apt
       for our purpose; please note that I am not referring to C++ namespaces
       or similiar programming language constructs.

.. [4] Local addresses are represented in HPX by the hpx::naming::address and
       hpx::naming::full_address class. 

.. [5] For IP (v4 and v6), the protocal specific address is the human-readable
       form of an IP address and a port.

.. [6] In HPX, the core component types of the runtime system
       (hpx::components::server::runtime_support, hpx::components::server::memory
       and hpx::components::server::memory_block at the time that this paper was
       written) register direct memory addresses with AGAS. Simple components
       also register direct addresses. Managed components, however, register a
       "virtual" address which must be further resolved with the hpx::get_lva()
       utility. 

