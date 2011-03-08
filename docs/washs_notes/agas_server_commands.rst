==================
 AGAS v1 Requests
==================
:author: Bryce Lelbach aka "wash"
:contact: blelbach@cct.lsu.edu
:organization: LSU Center for Computation and Technology, ParalleX group
:copyright: Copyright (c) 2011

Distributed under the Boost Software License, Version 1.0. (See accompanying 
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

------------
Introduction
------------

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

-----------------
command_getprefix
-----------------

::

  command_getprefix = 0,      ///< return a unique prefix for the requesting site

------------------------
command_getconsoleprefix
------------------------

::

  command_getconsoleprefix = 1, ///< return the prefix of the console locality

-----------------
command_getprefix
-----------------

::

  command_getprefixes = 2,    ///< return prefixes for all known localities in the system

-----------------
command_getprefix
-----------------

::

  command_getprefix_for_site = 3, ///< return prefix for the given site

-----------------
command_getprefix
-----------------

::

  command_get_component_id = 4,   ///< return an unique component type

-----------------
command_getprefix
-----------------

::

  command_register_factory = 5,   ///< register a factory for a component type

------------------
command_getidrange
------------------

::

  command_getidrange = 6,     ///< return a unique range of ids for the requesting site

------------------------------------------------------------
Local Address Resolution (resolve, unbind_range, bind_range)
------------------------------------------------------------

::

  command_bind_range = 7,     ///< bind a range of addresses to a range of global ids
  command_unbind_range = 10,  ///< remove binding for a range of global ids
  command_resolve = 11,       ///< resolve a global id to an address

:mappings: gid (hpx::naming::gid_type) -> local address (hpx::naming::address)

==================== ==============
AGAS v1 ptype        Method concept 
==================== ==============
command_resolve      resolve
command_bind_range   bind
command_unbind_range unbind
==================== ==============

---------------------------------------
GID Garbage Collection (incref, decref)
---------------------------------------

::

  command_incref = 8,         ///< increment global reference count for the given id
  command_decref = 9,         ///< decrement global reference count for the given id

--------------------------------------------------------
Namespace Resolution (queryid, registerid, unregisterid)
--------------------------------------------------------

::

  command_queryid = 12,       ///< query for a global id associated with a namespace name (string)
  command_registerid = 13,    ///< associate a namespace name with a global id
  command_unregisterid = 14,  ///< remove association of a namespace name with a global id

:mappings: name (std::string) -> gid (hpx::naming::gid_type)

==================== ==============
AGAS v1 ptype        Method concept 
==================== ==============
command_queryid      resolve
command_registerid   bind
command_unregisterid unbind
==================== ==============

The name of this request is not particularly concise; one might be led to
believe that this command unregisters a gid. This command is the unbind command
for the name --> gid service.

****************
Special Commands
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


