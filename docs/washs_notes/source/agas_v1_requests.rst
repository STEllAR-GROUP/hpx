==================
 AGAS v1 Requests
==================
:author: Bryce Lelbach aka "wash"
:contact: blelbach@cct.lsu.edu
:organization: LSU Center for Computation and Technology, ParalleX group
:copyright: Copyright (c) 2011

Distributed under the Boost Software License, Version 1.0. (See accompanying 
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

************
Introduction
************

*Disclaimer: Read my paper on the implementation of RTTI-free dynamic type
systems in C++ first and my introduction to AGAS.*

The backend of AGAS v1 is implemented in the hpx::naming::server::request_handler<>
class. This class implements all of the services that AGAS provides. Requests
are sent to the handler in the form of hpx::naming::server::request instances.

The request data structure holds two pieces of information:

  0) A punned type (ptype) from the bounded type system defined by
     hpx::naming::server::agas_server_command.
  1) Request data (note that not all ptypes have request data).

In AGAS v1, the C++ identifiers given to these types are a little bit ambiguous.
Unfortunately, these names are used by AGAS logging, so it is important for HPX
users to understand how this type system maps to the services provided by AGAS.

*************************
Requests to AGAS Services
*************************

----------------------------------------------------------------
Component/Factory Namespace (get_component_id, register_factory)
----------------------------------------------------------------

*Note: The component AGAS namespace and the factory AGAS namespace are tightly
coupled but distinct.*

::

  command_get_component_id = 4,   ///< return an unique component type
  command_register_factory = 5,   ///< register a factory for a component type

Mappings
^^^^^^^^

component name (std::string) -> type id (hpx::components::component_type) 
type id (hpx::components::component_type) -> prefixes (boost::uint32_t) [0]_

Interface
^^^^^^^^^

========================== ==============
AGAS v1 ptype              Method concept 
========================== ==============
command_get_component_id   resolve [1]_
command_register_factory   bind [2]_
========================== ==============

.. [0] This table currently provides no accessors, and is only used by the
       locality registration service (see command_getprefixes below).
.. [1] command_get_component_id resolves a component name to a type id.
.. [2] command_register_factory creates bindings in the component name ->
       type id database as well as the type id -> prefixes database. It is
       currently the only bind interface for this service.

---------------------------------------------------------------------------
Locality Namespace (getprefix, getprefixes, getprefix_for_site, getidrange)
---------------------------------------------------------------------------

::

  command_getprefix = 0,      ///< return a unique prefix for the requesting site
  command_getprefixes = 2,    ///< return prefixes for all known localities in the system
  command_getprefix_for_site = 3, ///< return prefix for the given site
  command_getidrange = 6,     ///< return a unique range of ids for the requesting site

Mappings
^^^^^^^^

locality (hpx::naming::locality) -> prefix (std::pair<boost::uint32_t, hpx::naming::gid_type>) [3]_

Interface
^^^^^^^^^

========================== ==============
AGAS v1 ptype              Method concept 
========================== ==============
command_getprefix          bind
command_getprefixes        resolve [4]_
command_getprefix_for_site resolve [5]_ 
command_getidrange         rebind
========================== ==============

.. [3] Note that locality registration uses the local address resolution
       service. This means that registering a locality will create bindings
       in both the locality -> prefix database and the gid -> address database.
.. [4] A command_getprefixes request is sent with a component type. If the
       component type is hpx::components::component_invalid, then a vector
       of all prefixes is returned with the reply. Otherwise, a vector of
       prefixes with have a factory for the given component type is returned.
       The factory lookup semantics provide the only direct resolution interface
       to the component type -> prefixes database provided by the factory
       registration service.  
.. [5] command_get_prefix_for_site actually just calls getprefix_for_site -
       the only different is that it doesn't try to verify that there's only
       one console locality.

-----------------------------------------------------
Primary Namespace (resolve, unbind_range, bind_range)
-----------------------------------------------------

::

  command_bind_range = 7,     ///< bind a range of addresses to a range of global ids
  command_unbind_range = 10,  ///< remove binding for a range of global ids
  command_resolve = 11,       ///< resolve a global id to an address

Mappings
^^^^^^^^

gid (hpx::naming::gid_type) -> local address (hpx::naming::address)

Interface
^^^^^^^^^

==================== ==============
AGAS v1 ptype        Method concept 
==================== ==============
command_resolve      resolve
command_bind_range   bind
command_unbind_range unbind
==================== ==============

-------------------------------------------------
GID Garbage Collection Namespace (incref, decref)
-------------------------------------------------

::

  command_incref = 8,         ///< increment global reference count for the given id
  command_decref = 9,         ///< decrement global reference count for the given id

----------------------------------------------------------------------
Symbol Namespace (queryid, registerid, unregisterid, getconsoleprefix)
----------------------------------------------------------------------

::

  command_getconsoleprefix = 1, ///< return the prefix of the console locality
  command_queryid = 12,       ///< query for a global id associated with a namespace name (string)
  command_registerid = 13,    ///< associate a namespace name with a global id
  command_unregisterid = 14,  ///< remove association of a namespace name with a global id

Mappings
^^^^^^^^

string (std::string) -> gid (hpx::naming::gid_type)

Interface
^^^^^^^^^

======================== ==============
AGAS v1 ptype            Method concept 
======================== ==============
command_getconsoleprefix resolve
command_queryid          resolve
command_registerid       bind
command_unregisterid     unbind 
======================== ==============

****************
Special Requests
****************

---------------
command_unknown
---------------

::

  command_unknown = -1,

This request is the "NULL" request. This is the default value of 
hpx::naming::server::request's constructor. A request holding a command value
of command_unknown is not valid; hpx::naming::server::request_handler<> will
reply to this command with hpx::bad_request.

--------------------
command_firstcommand
--------------------

::

  command_firstcommand = 0,

Syntactic sugar for the numerically lowest valid request value (which is
command_getprefix).

-------------------
command_lastcommand
-------------------

::

  command_lastcommand

Syntactic sugar for the numerically highest valid request value. Note that
unlike command_firstcommand, this is not inclusive (aka it's one-past-the-end).
command_lastcommand is therefore an invalid command itself. If it is sent to
hpx::naming::server::request_handler, AGAS will respond with hpx::bad_request.

-------------------------------------------------
command_statistics_count, command_statistics_mean
-------------------------------------------------

::

  command_statistics_count = 15,   ///< return some usage statistics: execution count 
  command_statistics_mean = 16,    ///< return some usage statistics: average server execution time

Self explanatory, likely uninteresting unless you are Hartmut. 

--------------------------
command_statistics_moment2
--------------------------

::

  command_statistics_moment2 = 17, ///< return some usage statistics: 2nd moment of server execution time

This is currently unimplemented, although instead of replying with
hpx::not_implemented, it simply returns 0.0.


