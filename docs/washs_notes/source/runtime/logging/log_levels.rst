=============
 HPX Logging 
=============
:author: Bryce Lelbach aka "wash"
:contact: blelbach@cct.lsu.edu
:organization: LSU Center for Computation and Technology, ParalleX group
:copyright: Copyright (c) 2011

Distributed under the Boost Software License, Version 1.0. (See accompanying 
file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

************
Introduction
************

HPX logging uses an older version of the Boost.Log library (a version which is,
unfortunately, incompatible with the version of Boost.Log which was accepted
into the Boost C++ libraries). These logs can be rather useful for debugging
HPX code. 

Note that the names of the ini sections associated with each category of logs
are a bit unintuitive. There are two configuration sections for each category,
of the form hpx.CATEGORY.logging and hpx.CATEGORY.logging.console. 

Further note that this document intentionally ignores the subject of customizing
HPX log formatting, in the interest of time and my own sanity.

------------
Destinations
------------

Logs will be saved to destinations specified in the main HPX ini file or by
environmental variables. By default, logs are saved to ./hpx.CATEGORY.PID.log
(where CATEGORY and PID are placeholders). The general log is saved to
./hpx.PID.log.

------
Levels
------

All HPX logs have seven different logging levels. These levels can be set in the
main HPX ini file, or through environmental variables (this is covered in more
detail later in this document).

The log levels and their associated integral values are shown in the table
below, ordered from most verbose to least verbose. By default, all HPX logs are
set to 0.

========== ==============
Log Level  Integral Value
========== ==============
<all>      6
<debug>    5
<info>     4
<warning>  3
<error>    2
<fatal>    1
No logging 0
========== ==============

---------------------------------
INI Environmental Variable Syntax
---------------------------------

In HPX ini files, the following syntax:::

  ${FOO:default}

Will use the environmental variable FOO if it is set and default otherwise. 
No default has to be specified. Therefore this:::

  ${FOO}

refers to the environmental variable FOO. If FOO is not set or empty the 
overall expression will evaluate to an empty string.

The syntax:::

  $[section.key:default]

refers to the value held by the 'key' in the ini 'section' if it exists and 
default otherwise. No default has to be specified. Therefore this:::

  $[section.key]

refers to the 'key' inside the given 'section'. If the key is not set or 
empty the overall expression will evaluate to an empty string.


*********
Reference
*********

------------
General Logs
------------

::

  [hpx.logging]
  level = ${HPX_LOGLEVEL:0}
  destination = ${HPX_LOGDESTINATION:console}

  [hpx.logging.console]
  level = ${HPX_LOGLEVEL:$[hpx.logging.level]}
  destination = ${HPX_CONSOLE_LOGDESTINATION:file(hpx.$[system.pid].log)}

-----------
Timing Logs
-----------

::

  [hpx.logging.timing]
  level = ${HPX_TIMING_LOGLEVEL:0}
  destination = ${HPX_TIMING_LOGDESTINATION:console}

  [hpx.logging.console.timing]
  level = ${HPX_TIMING_LOGLEVEL:$[hpx.logging.timing.level]}
  destination = ${HPX_CONSOLE_TIMING_LOGDESTINATION:file(hpx.timing.$[system.pid].log)}

---------
AGAS Logs
---------

::

  [hpx.logging.agas]
  level = ${HPX_AGAS_LOGLEVEL:0}
  destination = ${HPX_AGAS_LOGDESTINATION:file(hpx.agas.$[system.pid].log)}

  [hpx.logging.console.agas]
  level = ${HPX_AGAS_LOGLEVEL:$[hpx.logging.agas.level]}
  destination = ${HPX_CONSOLE_AGAS_LOGDESTINATION:file(hpx.agas.$[system.pid].log)}

----------------
Application Logs
----------------

::

  [hpx.logging.application]
  level = ${HPX_APP_LOGLEVEL:0}
  destination = ${HPX_APP_LOGDESTINATION:console}

  [hpx.logging.console.application]
  level = ${HPX_APP_LOGLEVEL:$[hpx.logging.application.level]}
  destination = ${HPX_CONSOLE_APP_LOGDESTINATION:file(hpx.application.$[system.pid].log)}

