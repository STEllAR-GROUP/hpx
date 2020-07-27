..
    Copyright (C) 2007-2017 Hartmut Kaiser
                  2014 University of Oregon

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

=============================
Optimizing |hpx| applications
=============================

.. _performance_counters:

Performance counters
====================

Performance counters in |hpx| are used to provide information as to how well the
runtime system or an application is performing. The counter data can help
determine system bottlenecks, and fine-tune system and application performance.
The |hpx| runtime system, its networking, and other layers provide counter data
that an application can consume to provide users with information about how
well the application is performing.

Applications can also use counter data to determine how much system resources to
consume. For example, an application that transfers data over the network could
consume counter data from a network switch to determine how much data to
transfer without competing for network bandwidth with other network traffic. The
application could use the counter data to adjust its transfer rate as the
bandwidth usage from other network traffic increases or decreases.

Performance counters are |hpx| parallel processes that expose a predefined
interface. |hpx| exposes special API functions that allow one to create, manage,
and read the counter data, and release instances of performance counters.
Performance Counter instances are accessed by name, and these names have a
predefined structure which is described in the section
:ref:`performance_counter_names`. The advantage of this is that any Performance
Counter can be accessed remotely (from a different :term:`locality`) or locally
(from the same :term:`locality`). Moreover, since all counters expose their data
using the same API, any code consuming counter data can be utilized to access
arbitrary system information with minimal effort.

Counter data may be accessed in real time. More information about how to consume
counter data can be found in the section :ref:`consuming`.

All |hpx| applications provide command line options related to performance
counters, such as the ability to list available counter types, or periodically
query specific counters to be printed to the screen or save them in a file. For
more information, please refer to the section :ref:`commandline`.

.. _performance_counter_names:

Performance counter names
-------------------------

All Performance Counter instances have a name uniquely identifying each
instance. This name can be used to access the counter, retrieve all related meta
data, and to query the counter data (as described in the section
:ref:`consuming`). Counter names are strings with a predefined structure. The
general form of a countername is:

.. code-block:: text

   /objectname{full_instancename}/countername@parameters

where ``full_instancename`` could be either another (full) counter name or a
string formatted as:

.. code-block:: text

   parentinstancename#parentindex/instancename#instanceindex

Each separate part of a countername (e.g., ``objectname``, ``countername``
``parentinstancename``, ``instancename``, and ``parameters``) should start with
a letter (``'a'``\ ...\ ``'z'``, ``'A'``\ ...\ ``'Z'``) or an underscore
character (``'_'``), optionally followed by letters, digits (``'0'``\ ...\
``'9'``), hyphen (``'-'``), or underscore characters. Whitespace is not allowed
inside a counter name. The characters ``'/'``, ``'{'``, ``'}'``, ``'#'`` and
``'@'`` have a special meaning and are used to delimit the different parts of
the counter name.

The parts ``parentinstanceindex`` and ``instanceindex`` are integers. If an
index is not specified, |hpx| will assume a default of ``-1``.

.. _example:

Two counter name examples
-------------------

This section gives examples of both simple counter names and aggregate
counter names. For more information on simple and aggregate counter
names, please see :ref:`performance_counter_instances`. 

An example of a well-formed (and meaningful) simple counter name would be:

.. code-block:: text

   /threads{locality#0/total}/count/cumulative

This counter returns the current cumulative number of executed (retired)
|hpx| threads for the :term:`locality` ``0``. The counter type of this counter
is ``/threads/count/cumulative`` and the full instance name is
``locality#0/total``. This counter type does not require an ``instanceindex`` or
``parameters`` to be specified.

In this case, the ``parentindex`` (the ``'0'``) designates the :term:`locality`
for which the counter instance is created. The counter will return the number of
|hpx| threads retired on that particular :term:`locality`.

Another example for a well formed (aggregate) counter name is:

.. code-block:: text

   /statistics{/threads{locality#0/total}/count/cumulative}/average@500

This counter takes the simple counter from the first example, samples its values
every ``500`` milliseconds, and returns the average of the value samples
whenever it is queried. The counter type of this counter is
``/statistics/average`` and the instance name is the full name of the counter
for which the values have to be averaged. In this case, the ``parameters`` (the
``'500'``) specify the sampling interval for the averaging to take place (in
milliseconds).

.. _types:

Performance counter types
-------------------------

Every performance counter belongs to a specific performance counter type which
classifies the counters into groups of common semantics. The type of a counter
is identified by the ``objectname`` and the ``countername`` parts of the name.

.. code-block:: text

   /objectname/countername

When an application starts |hpx| will register all available counter types on each of
the localities. These counter types are held in a special performance counter
registration database, which can be used to retrieve the meta data related
to a counter type and to create counter instances based on a given counter
instance name.

.. _instances:

Performance counter instances
-----------------------------

The ``full_instancename`` distinguishes different counter instances of the same
counter type. The formatting of the ``full_instancename`` depends on the counter
type. There are two types of counters: simple counters, which usually generate
the counter values based on direct measurements, and aggregate counters, which
take another counter and transform its values before generating their own
counter values. An example for a simple counter is given :ref:`above <example>`:
counting retired |hpx| threads. An aggregate counter is shown as an example
:ref:`above <example>` as well: calculating the average of the underlying
counter values sampled at constant time intervals.

While simple counters use instance names formatted as
``parentinstancename#parentindex/instancename#instanceindex``, most aggregate
counters have the full counter name of the embedded counter as their instance
name.

Not all simple counter types require specifying all four elements of a full counter
instance name; some of the parts (``parentinstancename``, ``parentindex``,
``instancename``, and ``instanceindex``) are optional for specific counters.
Please refer to the documentation of a particular counter for more information
about the formatting requirements for the name of this counter (see
:ref:`counters`).

The ``parameters`` are used to pass additional information to a counter at
creation time. They are optional, and they fully depend on the concrete counter.
Even if a specific counter type allows additional parameters to be given, those
usually are not required as sensible defaults will be chosen. Please refer to
the documentation of a particular counter for more information about what
parameters are supported, how to specify them, and what default values are
assumed (see also :ref:`counters`).

Every :term:`locality` of an application exposes its own set of performance
counter types and performance counter instances. The set of exposed counters is
determined dynamically at application start based on the execution environment
of the application. For instance, this set is influenced by the current hardware
environment for the :term:`locality` (such as whether the :term:`locality` has
access to accelerators), and the software environment of the application (such
as the number of OS threads used to execute |hpx| threads).

.. _wildcards:

Using wildcards in performance counter names
--------------------------------------------

It is possible to use wildcard characters when specifying performance counter
names. Performance counter names can contain two types of wildcard characters:

* Wildcard characters in the performance counter type
* Wildcard characters in the performance counter instance name

A wildcard character has a meaning which is very close to usual file name
wildcard matching rules implemented by common shells (like bash).

.. list-table:: Wildcard characters in the performance counter type

   * * Wildcard
     * Description
   * * ``*``
     * This wildcard character matches any number (zero or more) of arbitrary
       characters.
   * * ``?``
     * This wildcard character matches any single arbitrary character.
   * * ``[...]``
     * This wildcard character matches any single character from the list of
       specified within the square brackets.

.. list-table:: Wildcard characters in the performance counter instance name

   * * Wildcard
     * Description
   * * ``*``
     * This wildcard character matches any :term:`locality` or any thread,
       depending on whether it is used for ``locality#*`` or
       ``worker-thread#*``. No other wildcards are allowed in counter instance
       names.

.. _consuming:

Consuming performance counter data
----------------------------------

You can consume performance data using either the command line interface,
the |hpx| application or the |hpx| API. The command line interface is easier to
use, but it is less flexible and does not allow one to adjust the behaviour of
your application at runtime. The command line interface provides a convenience
abstraction but simplified abstraction for querying and logging performance
counter data for a set of performance counters.

.. _performance_counters_commandline:

Consuming performance counter data from the command line
--------------------------------------------------------

|hpx| provides a set of predefined command line options for every application
that uses ``hpx::init`` for its initialization. While there are many more
command line options available (see :ref:`commandline`), the set of options related
to performance counters allows one to list existing counters, and query existing
counters once at application termination or repeatedly after a constant time
interval.

The following table summarizes the available command line options:

.. list-table:: |hpx| Command Line Options Related to Performance Counters

   * * Command line option
     * Description
   * * ``--hpx:print-counter``
     * Prints the specified performance counter either repeatedly and/or at the
       times specified by ``--hpx:print-counter-at`` (see also option
       ``--hpx:print-counter-interval``).
   * * ``--hpx:print-counter-reset``
     * Prints the specified performance counter either repeatedly and/or at the
       times specified by ``--hpx:print-counter-at``. Reset the counter after the
       value is queried (see also option ``--hpx:print-counter-interval``).
   * * ``--hpx:print-counter-interval``
     * Prints the performance counter(s) specified with ``--hpx:print-counter``
       repeatedly after the time interval (specified in milliseconds)
       (default:``0`` which means print once at shutdown).
   * * ``--hpx:print-counter-destination``
     * Prints the performance counter(s) specified with ``--hpx:print-counter``
       to the given file (default: console).
   * * ``--hpx:list-counters``
     * Lists the names of all registered performance counters.
   * * ``--hpx:list-counter-infos``
     * Lists the description of all registered performance counters.
   * * ``--hpx:print-counter-format``
     * Prints the performance counter(s) specified with ``--hpx:print-counter``.
       Possible formats in CVS format with header or without any header (see
       option ``--hpx:no-csv-header``), possible values: ``csv`` (prints counter
       values in CSV format with full names as header) ``csv-short`` (prints
       counter values in CSV format with shortnames provided with
       ``--hpx:print-counter`` as ``--hpx:print-counter
       shortname,full-countername``).
   * * ``--hpx:no-csv-header``
     * Prints the performance counter(s) specified with ``--hpx:print-counter``
       and ``csv`` or ``csv-short`` format specified with
       ``--hpx:print-counter-format`` without header.
   * * ``--hpx:print-counter-at arg``
     * Prints the performance counter(s) specified with ``--hpx:print-counter``
       (or ``--hpx:print-counter-reset``) at the given point in time. Possible
       argument values: ``startup``, ``shutdown`` (default), ``noshutdown``.
   * * ``--hpx:reset-counters``
     * Resets all performance counter(s) specified with ``--hpx:print-counter``
       after they have been evaluated.
   * * ``--hpx:print-counter-types``
     * Appends counter type description to generated output.
   * * ``--hpx:print-counters-locally``
     * Each locality prints only its own local counters.

While the options ``--hpx:list-counters`` and ``--hpx:list-counter-infos`` give
a short list of all available counters, the full documentation for those can
be found in the section :ref:`counters`.

A simple example
----------------

All of the commandline options mentioned above can be tested using
the ``hello_world_distributed`` example.

Listing all available counters ``hello_world_distributed --hpx:list-counters``
yields:

.. code-block:: text

   List of available counter instances (replace * below with the appropriate
   sequence number)
   -------------------------------------------------------------------------
   /agas/count/allocate /agas/count/bind /agas/count/bind_gid
   /agas/count/bind_name ... /threads{locality#*/allocator#*}/count/objects
   /threads{locality#*/total}/count/stack-recycles
   /threads{locality#*/total}/idle-rate
   /threads{locality#*/worker-thread#*}/idle-rate

Providing more information about all available counters,
``hello_world_distributed --hpx:list-counter-infos`` yields:

.. code-block:: text

   Information about available counter instances (replace * below with the
   appropriate sequence number)
   ------------------------------------------------------------------------------
   fullname: /agas/count/allocate helptext: returns the number of invocations of
   the AGAS service 'allocate' type: counter_raw version: 1.0.0
   ------------------------------------------------------------------------------

   ------------------------------------------------------------------------------
   fullname: /agas/count/bind helptext: returns the number of invocations of the
   AGAS service 'bind' type: counter_raw version: 1.0.0
   ------------------------------------------------------------------------------

   ------------------------------------------------------------------------------
   fullname: /agas/count/bind_gid helptext: returns the number of invocations of
   the AGAS service 'bind_gid' type: counter_raw version: 1.0.0
   ------------------------------------------------------------------------------

   ...

This command will not only list the counter names but also a short description
of the data exposed by this counter.

.. note::

   The list of available counters may differ depending on the concrete execution
   environment (hardware or software) of your application.

Requesting the counter data for one or more performance counters can be achieved
by invoking ``hello_world_distributed`` with a list of counter names:

.. code-block:: bash

   hello_world_distributed \
       --hpx:print-counter=/threads{locality#0/total}/count/cumulative \
       --hpx:print-counter=/agas{locality#0/total}/count/bind

which yields for instance:

.. code-block:: bash

   hello world from OS-thread 0 on locality 0
   /threads{locality#0/total}/count/cumulative,1,0.212527,[s],33
   /agas{locality#0/total}/count/bind,1,0.212790,[s],11

The first line is the normal output generated by ``hello_world_distributed`` and
has no relation to the counter data listed. The last two lines contain the
counter data as gathered at application shutdown. These lines have six fields, the
counter name, the sequence number of the counter invocation, the time stamp at
which this information has been sampled, the unit of measure for the time stamp,
the actual counter value and an optional unit of measure for the counter value.

.. note::

   The command line option ``--hpx:print-counter-types` will append a seventh
   field to the generated output. This field will hold an abbreviated counter
   type.

The actual counter value can be represented by a single number (for counters
returning singular values) or a list of numbers separated by ``':'`` (for
counters returning an array of values, like for instance a histogram).

.. note::

   The name of the performance counter will be enclosed in double quotes ``'"'``
   if it contains one or more commas ``','``.

Requesting to query the counter data once after a constant time interval with
this command line:

.. code-block:: bash

   hello_world_distributed \
       --hpx:print-counter=/threads{locality#0/total}/count/cumulative \
       --hpx:print-counter=/agas{locality#0/total}/count/bind \
       --hpx:print-counter-interval=20

yields for instance (leaving off the actual console output of the
``hello_world_distributed`` example for brevity):

.. code-block:: text

   threads{locality#0/total}/count/cumulative,1,0.002409,[s],22
   agas{locality#0/total}/count/bind,1,0.002542,[s],9
   threads{locality#0/total}/count/cumulative,2,0.023002,[s],41
   agas{locality#0/total}/count/bind,2,0.023557,[s],10
   threads{locality#0/total}/count/cumulative,3,0.037514,[s],46
   agas{locality#0/total}/count/bind,3,0.038679,[s],10

The command ``--hpx:print-counter-destination=<file>`` will redirect all counter
data gathered to the specified file name, which avoids cluttering the console
output of your application.

The command line option ``--hpx:print-counter`` supports using a limited set of
wildcards for a (very limited) set of use cases. In particular, all occurrences
of ``#*`` as in ``locality#*`` and in ``worker-thread#*`` will be automatically
expanded to the proper set of performance counter names representing the actual
environment for the executed program. For instance, if your program is utilizing
four worker threads for the execution of |hpx| threads (see command line option
:option:`--hpx:threads`) the following command line

.. code-block:: bash

   hello_world_distributed \
       --hpx:threads=4 \
       --hpx:print-counter=/threads{locality#0/worker-thread#*}/count/cumulative

will print the value of the performance counters monitoring each of the worker
threads:

.. code-block:: text

   hello world from OS-thread 1 on locality 0
   hello world from OS-thread 0 on locality 0
   hello world from OS-thread 3 on locality 0
   hello world from OS-thread 2 on locality 0
   /threads{locality#0/worker-thread#0}/count/cumulative,1,0.0025214,[s],27
   /threads{locality#0/worker-thread#1}/count/cumulative,1,0.0025453,[s],33
   /threads{locality#0/worker-thread#2}/count/cumulative,1,0.0025683,[s],29
   /threads{locality#0/worker-thread#3}/count/cumulative,1,0.0025904,[s],33

The command ``--hpx:print-counter-format`` takes values ``csv`` and
``csv-short`` to generate CSV formatted counter values with a header.

With format as csv:

.. code-block:: bash

   hello_world_distributed \
       --hpx:threads=2 \
       --hpx:print-counter-format csv \
       --hpx:print-counter /threads{locality#*/total}/count/cumulative \
       --hpx:print-counter /threads{locality#*/total}/count/cumulative-phases

will print the values of performance counters in CSV format with the full
countername as a header:

.. code-block:: text

   hello world from OS-thread 1 on locality 0
   hello world from OS-thread 0 on locality 0
   /threads{locality#*/total}/count/cumulative,/threads{locality#*/total}/count/cumulative-phases
   39,93

With format csv-short:

.. code-block:: bash

   hello_world_distributed \
       --hpx:threads 2 \
       --hpx:print-counter-format csv-short \
       --hpx:print-counter cumulative,/threads{locality#*/total}/count/cumulative \
       --hpx:print-counter phases,/threads{locality#*/total}/count/cumulative-phases

will print the values of performance counters in CSV format with the short countername
as a header:

.. code-block:: text

   hello world from OS-thread 1 on locality 0
   hello world from OS-thread 0 on locality 0
   cumulative,phases
   39,93

With format csv and csv-short when used with ``--hpx:print-counter-interval``:

.. code-block:: bash

   hello_world_distributed \
       --hpx:threads 2 \
       --hpx:print-counter-format csv-short \
       --hpx:print-counter cumulative,/threads{locality#*/total}/count/cumulative \
       --hpx:print-counter phases,/threads{locality#*/total}/count/cumulative-phases \
       --hpx:print-counter-interval 5

will print the header only once repeating the performance counter value(s) repeatedly:

.. code-block:: text

   cum,phases
   25,42
   hello world from OS-thread 1 on locality 0
   hello world from OS-thread 0 on locality 0
   44,95

The command ``--hpx:no-csv-header`` can be used with
``--hpx:print-counter-format`` to print performance counter values in CSV format
without any header:

.. code-block:: bash

   hello_world_distributed \
   --hpx:threads 2 \
   --hpx:print-counter-format csv-short \
   --hpx:print-counter cumulative,/threads{locality#*/total}/count/cumulative \
   --hpx:print-counter phases,/threads{locality#*/total}/count/cumulative-phases \
   --hpx:no-csv-header

will print:

.. code-block:: text

   hello world from OS-thread 1 on locality 0
   hello world from OS-thread 0 on locality 0
   37,91

.. _api:

Consuming performance counter data using the |hpx| API
------------------------------------------------------

|hpx| provides an API that allows users to discover performance counters and
to retrieve the current value of any existing performance counter from any application.

Discover existing performance counters
--------------------------------------

Retrieve the current value of any performance counter
-----------------------------------------------------

Performance counters are specialized |hpx| components. In order to retrieve a
counter value, the performance counter needs to be instantiated. |hpx| exposes a
client component object for this purpose::

    hpx::performance_counters::performance_counter counter(std::string const& name);

Instantiating an instance of this type will create the performance counter
identified by the given ``name``. Only the first invocation for any given counter
name will create a new instance of that counter. All following invocations for a
given counter name will reference the initially created instance. This ensures
that at any point in time there is never more than one active instance of
any of the existing performance counters.

In order to access the counter value (or to invoke any of the other functionality
related to a performance counter, like ``start``, ``stop`` or ``reset``) member
functions of the created client component instance should be called::

    // print the current number of threads created on locality 0
    hpx::performance_counters::performance_counter count(
        "/threads{locality#0/total}/count/cumulative");
    hpx::cout << count.get_value<int>().get() << hpx::endl;

For more information about the client component type, see |:cpp:class:`hpx::performance_counters::performance_counter` 

.. note::

   In the above example ``count.get_value()`` returns a future. In order to print
   the result we must append ``.get()`` to retrieve the value. You could write the
   above example like this for more clarity::

       // print the current number of threads created on locality 0
       hpx::performance_counters::performance_counter count(
           "/threads{locality#0/total}/count/cumulative");
       hpx::future<int> result = count.get_value<int>();
       hpx::cout << result.get() << hpx::endl;

.. _providing:

Providing performance counter data
----------------------------------

|hpx| offers several ways by which you may provide your own data as a
performance counter. This has the benefit of exposing additional, possibly
application-specific information using the existing Performance Counter
framework, unifying the process of gathering data about your application.

An application that wants to provide counter data can implement a performance
counter to provide the data. When a consumer queries performance data, the |hpx|
runtime system calls the provider to collect the data. The runtime system uses
an internal registry to determine which provider to call.

Generally, there are two ways of exposing your own performance counter data: a
simple, function-based way and a more complex, but more powerful way of
implementing a full performance counter. Both alternatives are described in the
following sections.

.. _simple_counters:

Exposing performance counter data using a simple function
---------------------------------------------------------

The simplest way to expose arbitrary numeric data is to write a function which
will then be called whenever a consumer queries this counter. Currently, this
type of performance counter can only be used to expose integer values. The
expected signature of this function is::

    std::int64_t some_performance_data(bool reset);

The argument ``bool reset`` (which is supplied by the runtime system when the
function is invoked) specifies whether the counter value should be reset after
evaluating the current value (if applicable).

For instance, here is such a function returning how often it was invoked::

    // The atomic variable 'counter' ensures the thread safety of the counter.
    boost::atomic<std::int64_t> counter(0);

    std::int64_t some_performance_data(bool reset)
    {
        std::int64_t result = ++counter;
        if (reset)
            counter = 0;
        return result;
    }

This example function exposes a linearly-increasing value as our performance
data. The value is incremented on each invocation, i.e., each time a consumer
requests the counter data of this performance counter.

The next step in exposing this counter to the runtime system is to register the
function as a new raw counter type using the |hpx| API function
:cpp:func:`hpx::performance_counters::install_counter_type`. A counter type
represents certain common characteristics of counters, like their counter type
name and any associated description information. The following snippet shows an
example of how to register the function ``some_performance_data``, which is shown
above, for a counter type named ``"/test/data"``. This registration has to be
executed before any consumer instantiates, and queries an instance of this
counter type::

    #include <hpx/include/performance_counters.hpp>

    void register_counter_type()
    {
        // Call the HPX API function to register the counter type.
        hpx::performance_counters::install_counter_type(
            "/test/data",                                   // counter type name
            &some_performance_data,                         // function providing counter data
            "returns a linearly increasing counter value"   // description text (optional)
            ""                                              // unit of measure (optional)
        );
    }

Now it is possible to instantiate a new counter instance based on the naming
scheme ``"/test{locality#*/total}/data"`` where ``*`` is a zero-based integer
index identifying the :term:`locality` for which the counter instance should be
accessed. The function
:cpp:func:`hpx::performance_counters::install_counter_type` enables users to
instantiate exactly one counter instance for each :term:`locality`. Repeated
requests to instantiate such a counter will return the same instance, i.e., the
instance created for the first request.

If this counter needs to be accessed using the standard |hpx| command line
options, the registration has to be performed during application startup, before
``hpx_main`` is executed. The best way to achieve this is to register an |hpx|
startup function using the API function
:cpp:func:`hpx::register_startup_function` before calling ``hpx::init`` to
initialize the runtime system::

    int main(int argc, char* argv[])
    {
        // By registering the counter type we make it available to any consumer
        // who creates and queries an instance of the type "/test/data".
        //
        // This registration should be performed during startup. The
        // function 'register_counter_type' should be executed as an HPX thread right
        // before hpx_main is executed.
        hpx::register_startup_function(&register_counter_type);

        // Initialize and run HPX.
        return hpx::init(argc, argv);
    }

Please see the code in :download:`simplest_performance_counter.cpp <../../examples/performance_counters/simplest_performance_counter.cpp>`
for a full example demonstrating this functionality.

.. _full_counters:

Implementing a full performance counter
---------------------------------------

Sometimes, the simple way of exposing a single value as a performance counter is
not sufficient. For that reason, |hpx| provides a means of implementing full
performance counters which support:

* Retrieving the descriptive information about the performance counter
* Retrieving the current counter value
* Resetting the performance counter (value)
* Starting the performance counter
* Stopping the performance counter
* Setting the (initial) value of the performance counter

Every full performance counter will implement a predefined interface:

.. literalinclude:: ../../libs/performance_counters/include/hpx/performance_counters/performance_counter.hpp
   :language: c++

In order to implement a full performance counter, you have to create an |hpx|
component exposing this interface. To simplify this task, |hpx| provides a
ready-made base class which handles all the boiler plate of creating
a component for you. The remainder of this section will explain the process of creating a full
performance counter based on the Sine example, which you can find in the
directory ``examples/performance_counters/sine/``.

The base class is defined in the header file [hpx_link
hpx/performance_counters/base_performance_counter.hpp..hpx/performance_counters/base_performance_counter.hpp]
as:

.. literalinclude:: ../../libs/performance_counters/include/hpx/performance_counters/base_performance_counter.hpp
   :language: c++

The single template parameter is expected to receive the type of the
derived class implementing the performance counter. In the Sine example
this looks like:

.. literalinclude:: ../../examples/performance_counters/sine/server/sine.hpp
   :language: c++

i.e., the type ``sine_counter`` is derived from the base class passing the type
as a template argument (please see :download:`simplest_performance_counter.cpp <../../examples/performance_counters/simplest_performance_counter.cpp>`
for the full source code of the counter definition). For more information about this
technique (called Curiously Recurring Template Pattern - CRTP), please see for
instance the corresponding `Wikipedia article
<http://en.wikipedia.org/wiki/Curiously_recurring_template_pattern>`_. This base
class itself is derived from the ``performance_counter`` interface described
above.

Additionally, a full performance counter implementation not only exposes the
actual value but also provides information about:

* The point in time a particular value was retrieved.
* A (sequential) invocation count.
* The actual counter value.
* An optional scaling coefficient.
* Information about the counter status.

.. _counters:

Existing |hpx| performance counters
-----------------------------------

The |hpx| runtime system exposes a wide variety of predefined performance
counters. These counters expose critical information about different modules of
the runtime system. They can help determine system bottlenecks and fine-tune
system and application performance.

.. list-table:: :term:`AGAS` performance counters

   * * Counter type
     * Counter instance formatting
     * Description
     * Parameters
   * * ``/agas/count/<agas_service>``

       where:

       ``<agas_service>`` is one of the following:

       *primary namespace services*: ``route``, ``bind_gid``, ``resolve_gid``,
       ``unbind_gid``, ``increment_credit``, ``decrement_credit``, ``allocate``,
       ``begin_migration``, ``end_migration``

       *component namespace services*: ``bind_prefix``, ``bind_name``,
       ``resolve_id``, ``unbind_name``, ``iterate_types``,
       ``get_component_typename``, ``num_localities_type``

       *locality namespace services*: ``free``, ``localities``,
       ``num_localities``, ``num_threads``, ``resolve_locality``,
       ``resolved_localities``

       *symbol namespace services*: ``bind``, ``resolve``, ``unbind``,
       ``iterate_names``, ``on_symbol_namespace_event``
     * ``<agas_instance>/total``

       where:

       ``<agas_instance>`` is the name of the :term:`AGAS` service to query.
       Currently, this value will be ``locality#0`` where ``0`` is the root
       :term:`locality` (the id of the locality hosting the :term:`AGAS`
       service).

       The value for ``*`` can be any :term:`locality` id for the following
       ``<agas_service>``: ``route``, ``bind_gid``, ``resolve_gid``,
       ``unbind_gid``, ``increment_credit``, ``decrement_credit``, ``bin``,
       ``resolve``, ``unbind``, and ``iterate_names`` (only the primary and
       symbol :term:`AGAS` service components live on all localities, whereas
       all other :term:`AGAS` services are available on ``locality#0`` only).
     * None
     * Returns the total number of invocations of the specified :term:`AGAS`
       service since its creation.
   * * ``/agas/<agas_service_category>/count``

       where:

       ``<agas_service_category>`` is one of the following: ``primary``,
       ``locality``, ``component`` or ``symbol``

     * ``<agas_instance>/total``

       where:

       ``<agas_instance>`` is the name of the :term:`AGAS` service to query.
       Currently, this value will be ``locality#0`` where ``0`` is the root
       :term:`locality` (the id of the :term:`locality` hosting the :term:`AGAS`
       service). Except for ``<agas_service_category>``, ``primary`` or
       ``symbol`` for which the value for ``*`` can be any :term:`locality` id
       (only the primary and symbol :term:`AGAS` service components live on all
       localities, whereas all other :term:`AGAS` services are available on
       ``locality#0`` only).
     * None
     * Returns the overall total number of invocations of all :term:`AGAS`
       services provided by the given :term:`AGAS` service category since its
       creation.
   * * ``agas/time/<agas_service>``

       where:

       ``<agas_service>`` is one of the following:


       *primary namespace services*: ``route``, ``bind_gid``, ``resolve_gid``,
       ``unbind_gid``, ``increment_credit``, ``decrement_credit``, ``allocate``
       ``begin_migration``, ``end_migration``

       *component namespace services*: ``bind_prefix``, ``bind_name``,
       ``resolve_id``, ``unbind_name``, ``iterate_types``,
       ``get_component_typename``, ``num_localities_type``

       *locality namespace services*: ``free``, ``localities``,
       ``num_localities``, ``num_threads``, ``resolve_locality``,
       ``resolved_localities``

       *symbol namespace services*: ``bind``, ``resolve``, ``unbind``,
       ``iterate_names``, ``on_symbol_namespace_event``

     * ``<agas_instance>/total``

       where:

       ``<agas_instance>`` is the name of the :term:`AGAS` service to query.
       Currently, this value will be ``locality#0`` where ``0`` is the root
       :term:`locality` (the id of the :term:`locality` hosting the :term:`AGAS`
       service).

       The value for ``*`` can be any :term:`locality` id for the following
       ``<agas_service>``: ``route``, ``bind_gid``, ``resolve_gid``,
       ``unbind_gid``, ``increment_credit``, ``decrement_credit``, ``bin``,
       ``resolve``, ``unbind``, and ``iterate_names`` (only the primary and
       symbol :term:`AGAS` service components live on all localities, whereas
       all other :term:`AGAS` services are available on ``locality#0`` only).
     * None
     * Returns the overall execution time of the specified :term:`AGAS` service
       since its creation (in nanoseconds).
   * * ``/agas/<agas_service_category>/time``

       where:

       ``<agas_service_category>`` is one of the following: ``primary``,
       ``locality``, ``component`` or ``symbol``
     * ``<agas_instance>/total``

       where:

       ``<agas_instance>`` is the name of the :term:`AGAS` service to query.
       Currently, this value will be ``locality#0`` where ``0`` is the root
       :term:`locality` (the id of the :term:`locality` hosting the :term:`AGAS`
       service). Except for ``<agas_service_category`` ``primary`` or ``symbol``
       for which the value for ``*`` can be any :term:`locality` id (only the
       primary and symbol :term:`AGAS` service components live on all
       localities, whereas all other :term:`AGAS` services are available on
       ``locality#0`` only).
     * None
     * Returns the overall execution time of all :term:`AGAS` services provided
       by the given :term:`AGAS` service category since its creation (in
       nanoseconds).
   * * ``/agas/count/entries``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the :term:`AGAS`
       cache should be queried. The :term:`locality` id is a (zero based) number
       identifying the :term:`locality`.
     * None
     * Returns the number of cache entries resident in the :term:`AGAS` cache of
       the specified :term:`locality` (see ``<cache_statistics>``).
   * * ``/agas/count/<cache_statistics>``

       where:

       ``<cache_statistics>`` is one of the following: ``cache/evictions``,
       ``cache/hits``, ``cache/insertions``, ``cache/misses``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the :term:`AGAS`
       cache should be queried. The :term:`locality` id is a (zero based) number
       identifying the :term:`locality`.
     * None
     * Returns the number of cache events (evictions, hits, inserts, and misses)
       in the :term:`AGAS` cache of the specified :term:`locality` (see
       ``<cache_statistics>``).
   * * ``/agas/count/<full_cache_statistics>``

       where:

       ``<full_cache_statistics>`` is one of the following: ``cache/get_entry``,
       ``cache/insert_entry``, ``cache/update_entry``, ``cache/erase_entry``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the :term:`AGAS`
       cache should be queried. The :term:`locality` id is a (zero based) number
       identifying the :term:`locality`.
     * None
     * Returns the number of invocations of the specified cache API function of
       the :term:`AGAS` cache.
   * * ``/agas/time/<full_cache_statistics>``

       where:

       ``<full_cache_statistics>`` is one of the following:
       ``cache/get_entry``, ``cache/insert_entry``, ``cache/update_entry``,
       ``cache/erase_entry``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the :term:`AGAS`
       cache should be queried. The :term:`locality` id is a (zero based) number
       identifying the :term:`locality`.
     * None
     * Returns the overall time spent executing of the specified API function of
       the :term:`AGAS` cache.

.. list-table:: :term:`Parcel` layer performance counters

   * * Counter type
     * Counter instance formatting
     * Description
     * Parameters
   * * ``/data/count/<connection_type>/<operation>``

       where:

       ``<operation>`` is one of the following: ``sent``, ``received``

       ``<connection_type`` is one of the following: ``tcp``, ``mpi``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the overall
       number of transmitted bytes should be queried for. The :term:`locality`
       id is a (zero based) number identifying the :term:`locality`.
     * Returns the overall number of raw (uncompressed) bytes sent or received
       (see ``<operation``, e.g. ``en`` or ``eceived``) for the specified
       ``<connection_type>``.

       The performance counters for the connection type ``mpi`` are available
       only if the compile time constant ``HPX_HAVE_PARCELPORT_MPI`` was defined
       while compiling the |hpx| core library (which is not defined by default,
       the corresponding cmake configuration constant is
       ``HPX_WITH_PARCELPORT_MPI``.

       Please see :ref:`cmake_variables` for more details.
     * None
   * * ``/data/time/<connection_type>/<operation>``

       where:

       ``<operation>`` is one of the following: ``sent``, ``received``

       ``<connection_type`` is one of the following: ``tcp``, ``mpi``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the total
       transmission time should be queried for. The :term:`locality` id is a
       (zero based) number identifying the :term:`locality`.
     * Returns the total time (in nanoseconds) between the start of each
       asynchronous transmission operation and the end of the corresponding
       operation for the specified ``<connection_type>`` the given
       :term:`locality` (see ``<operation``, e.g. ``en`` or ``eceived``).

       The performance counters for the connection type ``mpi`` are available
       only if the compile time constant ``HPX_HAVE_PARCELPORT_MPI`` was defined
       while compiling the |hpx| core library (which is not defined by default,
       the corresponding cmake configuration constant is
       ``HPX_WITH_PARCELPORT_MPI``.

       Please see :ref:`cmake_variables` for more details.
     * None
   * * ``/serialize/count/<connection_type>/<operation>``

       where:

       ``<operation>`` is one of the following: ``sent``, ``received``

       ``<connection_type`` is one of the following: ``tcp``, ``mpi``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the overall
       number of transmitted bytes should be queried for. The :term:`locality`
       id is a (zero based) number identifying the :term:`locality`.
     * Returns the overall number of bytes transferred (see ``<operation>``,
       e.g. ``sent`` or ``received`` possibly compressed) for the specified
       ``<connection_type>`` by the given :term:`locality`.

       The performance counters for the connection type ``mpi`` are available
       only if the compile time constant ``HPX_HAVE_PARCELPORT_MPI`` was defined
       while compiling the |hpx| core library (which is not defined by default,
       the corresponding cmake configuration constant is
       ``HPX_WITH_PARCELPORT_MPI``.

       Please see :ref:`cmake_variables` for more details.
     * If the configure-time option ``-DHPX_WITH_PARCELPORT_ACTION_COUNTERS=On``
       was specified, this counter allows one to specify an optional action name
       as its parameter. In this case the counter will report the number of
       bytes transmitted for the given action only.
   * * ``/serialize/time/<connection_type>/<operation>``

       where:

       ``<operation>`` is one of the following: ``sent``, ``received``

       ``<connection_type`` is one of the following: ``tcp``, ``mpi``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the
       serialization time should be queried for. The :term:`locality` id is a
       (zero based) number identifying the :term:`locality`.
     * Returns the overall time spent performing outgoing data serialization for
       the specified ``<connection_type>`` on the given :term:`locality` (see
       ``<operation``, e.g. ``sent`` or ``received``).

       The performance counters for the connection type ``mpi`` are available
       only if the compile time constant ``HPX_HAVE_PARCELPORT_MPI`` was defined
       while compiling the |hpx| core library (which is not defined by default,
       the corresponding cmake configuration constant is
       ``HPX_WITH_PARCELPORT_MPI``.

       Please see :ref:`cmake_variables` for more details.
     * If the configure-time option ``-DHPX_WITH_PARCELPORT_ACTION_COUNTERS=On``
       was specified, this counter allows one to specify an optional action name
       as its parameter. In this case the counter will report the serialization
       time for the given action only.
   * * ``/parcels/count/routed``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the number of
       routed parcels should be queried for. The :term:`locality` id is a (zero
       based) number identifying the :term:`locality`.
     * Returns the overall number of routed (outbound) parcels transferred by
       the given :term:`locality`.

       Routed parcels are those which cannot directly be delivered to its
       destination as the local :term:`AGAS` is not able to resolve the
       destination address. In this case a parcel is sent to the :term:`AGAS`
       service component which is responsible for creating the destination GID
       (and is responsible for resolving the destination address). This
       :term:`AGAS` service component will deliver the parcel to its final
       target.
     * If the configure-time option ``-DHPX_WITH_PARCELPORT_ACTION_COUNTERS=On``
       was specified, this counter allows one to specify an optional action name
       as its parameter. In this case the counter will report the number of
       parcels for the given action only.
   * * ``/parcels/count/<connection_type>/<operation>``

       where:

       ``<operation>`` is one of the following: ``sent``, ``received``

       ``<connection_type`` is one of the following: ``tcp``, ``mpi``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the number of
       parcels should be queried for. The :term:`locality` id is a (zero based)
       number identifying the :term:`locality`.
     * Returns the overall number of parcels transferred using the specified
       ``<connection_type`` by the given :term:`locality` (see ``operation>``,
       e.g. ``sent`` or ``received``.

       The performance counters for the connection type ``mpi`` are available
       only if the compile time constant ``HPX_HAVE_PARCELPORT_MPI`` was defined
       while compiling the |hpx| core library (which is not defined by default,
       the corresponding cmake configuration constant is
       ``HPX_WITH_PARCELPORT_MPI``.

       Please see :ref:`cmake_variables` for more details.
     * None
   * * ``/messages/count/<connection_type>/<operation>``

       where:

       ``<operation>`` is one of the following: ``sent``, ``received``

       ``<connection_type`` is one of the following: ``tcp``, ``mpi``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the number of
       messages should be queried for. The :term:`locality` id is a (zero based)
       number identifying the :term:`locality`.
     * Returns the overall number of messages [#]_ transferred using the
       specified ``<connection_type>`` by the given :term:`locality` (see
       ``<operation``, e.g. ``sent`` or ``received``)

       The performance counters for the connection type ``mpi`` are available
       only if the compile time constant ``HPX_HAVE_PARCELPORT_MPI`` was defined
       while compiling the |hpx| core library (which is not defined by default,
       the corresponding cmake configuration constant is
       ``HPX_WITH_PARCELPORT_MPI``.

       Please see :ref:`cmake_variables` for more details.
     * None
   * * ``/parcelport/count/<connection_type>/<cache_statistics>``

       where:

       ``<cache_statistics>`` is one of the following: ``cache/insertions``,
       ``cache/evictions``, ``cache/hits``, ``cache/misses``

       `<connection_type`` is one of the following: ``tcp``, ``mpi``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the number of
       messages should be queried for. The :term:`locality` id is a (zero based)
       number identifying the :term:`locality`.
     * Returns the overall number cache events (evictions, hits, inserts,
       misses, and reclaims) for the connection cache of the given connection
       type on the given :term:`locality` (see ``<cache_statistics``, e.g.
       ``ache/insertions``, ``cache/evictions``, ``cache/hits``,
       ``cache/misses`` or``cache/reclaims``.

       The performance counters for the connection type ``mpi`` are available
       only if the compile time constant ``HPX_HAVE_PARCELPORT_MPI`` was defined
       while compiling the |hpx| core library (which is not defined by default,
       the corresponding cmake configuration constant is
       ``HPX_WITH_PARCELPORT_MPI``.

       Please see :ref:`cmake_variables` for more details.
     * None
   * * ``/parcelqueue/length/<operation>``

       where:

       ``<operation>`` is one of the following: ``send``, ``receive``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the :term:`parcel` queue
       should be queried. The :term:`locality` id is a (zero based) number
       identifying the :term:`locality`.
     * Returns the current number of parcels stored in the :term:`parcel` queue (see
       ``<operation`` for which queue to query, e.g. ``sent`` or ``received``).
     * None

.. list-table:: Thread manager performance counters

   * * Counter type
     * Counter instance formatting
     * Description
     * Parameters
   * * ``/threads/count/cumulative``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the overall
       number of retired |hpx|-threads should be queried for. The
       :term:`locality` id (given by ``*`` is a (zero based) number identifying
       the :term:`locality`.

       ``pool#*`` is defining the pool for which the current value of the
        idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the overall
        number of retired |hpx|-threads should be queried for. The worker thread
        number (given by the ``*`` is a (zero based) number identifying the
        worker thread. The number of available worker threads is usually
        specified on the command line for the application using the option
        :option:`--hpx:threads`. If no pool-name is specified the counter refers
        to the 'default' pool.
     * Returns the overall number of executed (retired) |hpx|-threads on the
       given :term:`locality` since application start. If the instance name is
       ``total`` the counter returns the accumulated number of retired
       |hpx|-threads for all worker threads (cores) on that :term:`locality`. If
       the instance name is ``worker-thread#*`` the counter will return the
       overall number of retired |hpx|-threads for all worker threads
       separately. This counter is available only if the configuration time
       constant ``HPX_WITH_THREAD_CUMULATIVE_COUNTS`` is set to ``ON`` (default:
       ``ON``).
     * None
   * * ``/threads/time/average``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or


       ``locality#*/pool#*/worker-thread#*``

       where:


       ``locality#*`` is defining the :term:`locality` for which the average
       time spent executing one |hpx|-thread should be queried for. The
       :term:`locality` id (given by ``*`` is a (zero based) number identifying
       the :term:`locality`.

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the average
       time spent executing one |hpx|-thread should be queried for. The worker
       thread number (given by the ``*`` is a (zero based) number identifying
       the worker thread. The number of available worker threads is usually
       specified on the command line for the application using the option
       :option:`--hpx:threads`. If no pool-name is specified the counter refers
       to the 'default' pool.
     * Returns the average time spent executing one |hpx|-thread on the given
       :term:`locality` since application start. If the instance name is ``total``
       the counter returns the average time spent executing one |hpx|-thread for
       all worker threads (cores) on that :term:`locality`. If the instance name
       is ``worker-thread#*`` the counter will return the average time spent
       executing one |hpx|-thread for all worker threads separately. This counter
       is available only if the configuration time constants
       ``HPX_WITH_THREAD_CUMULATIVE_COUNTS`` (default: ``ON``) and
       ``HPX_WITH_THREAD_IDLE_RATES`` are set to ``ON`` (default: ``OFF``). The unit
       of measure for this counter is nanosecond [ns].
     * None
   * * ``/threads/time/average-overhead``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the average
       overhead spent executing one |hpx|-thread should be queried for. The
       :term:`locality` id (given by ``*`` is a (zero based) number identifying
       the :term:`locality`.

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the average
       overhead spent executing one |hpx|-thread should be queried for. The
       worker thread number (given by the ``*`` is a (zero based) number
       identifying the worker thread. The number of available worker threads is
       usually specified on the command line for the application using the
       option :option:`--hpx:threads`. If no pool-name is specified the counter
       refers to the 'default' pool.
     * Returns the average time spent on overhead while executing one
       |hpx|-thread on the given :term:`locality` since application start. If
       the instance name is ``total`` the counter returns the average time spent
       on overhead while executing one |hpx|-thread for all worker threads
       (cores) on that :term:`locality`. If the instance name is
       ``worker-thread#*`` the counter will return the average time spent on
       overhead executing one |hpx|-thread for all worker threads separately.
       This counter is available only if the configuration time constants
       ``HPX_WITH_THREAD_CUMULATIVE_COUNTS`` (default: ``ON``) and
       ``HPX_WITH_THREAD_IDLE_RATES`` are set to ``ON`` (default: ``OFF``). The
       unit of measure for this counter is nanosecond [ns].
     * None
   * * ``/threads/count/cumulative-phases``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the overall
       number of executed |hpx|-thread phases (invocations) should be queried
       for. The :term:`locality` id (given by ``*`` is a (zero based) number
       identifying the :term:`locality`.

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the overall
       number of executed |hpx|-thread phases (invocations) should be queried
       for. The worker thread number (given by the ``*`` is a (zero based)
       number identifying the worker thread. The number of available worker
       threads is usually specified on the command line for the application
       using the option :option:`--hpx:threads`. If no pool-name is specified
       the counter refers to the 'default' pool.
     * Returns the overall number of executed |hpx|-thread phases (invocations)
       on the given :term:`locality` since application start. If the instance
       name is ``total`` the counter returns the accumulated number of executed
       |hpx|-thread phases (invocations) for all worker threads (cores) on that
       :term:`locality`. If the instance name is ``worker-thread#*`` the counter
       will return the overall number of executed |hpx|-thread phases for all
       worker threads separately. This counter is available only if the
       configuration time constant ``HPX_WITH_THREAD_CUMULATIVE_COUNTS`` is set
       to ``ON`` (default: ``ON``). The unit of measure for this counter is
       nanosecond [ns].
     * None
   * * ``/threads/time/average-phase``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the average
       time spent executing one |hpx|-thread phase (invocation) should be
       queried for. The :term:`locality` id (given by ``*`` is a (zero based)
       number identifying the :term:`locality`.

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the average
       time executing one |hpx|-thread phase (invocation) should be queried for.
       The worker thread number (given by the ``*`` is a (zero based) number
       identifying the worker thread. The number of available worker threads is
       usually specified on the command line for the application using the
       option :option:`--hpx:threads`. If no pool-name is specified the counter
       refers to the 'default' pool.
     * Returns the average time spent executing one |hpx|-thread phase
       (invocation) on the given :term:`locality` since application start. If
       the instance name is ``total`` the counter returns the average time spent
       executing one |hpx|-thread phase (invocation) for all worker threads
       (cores) on that :term:`locality`. If the instance name is
       ``worker-thread#*`` the counter will return the average time spent
       executing one |hpx|-thread phase for all worker threads separately. This
       counter is available only if the configuration time constants
       ``HPX_WITH_THREAD_CUMULATIVE_COUNTS`` (default: ``ON``) and
       ``HPX_WITH_THREAD_IDLE_RATES`` are set to ``ON`` (default: ``OFF``). The
       unit of measure for this counter is nanosecond [ns].
     * None
   * * ``/threads/time/average-phase-overhead``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:


       ``locality#*`` is defining the :term:`locality` for which the average
       time overhead executing one |hpx|-thread phase (invocation) should be
       queried for. The :term:`locality` id (given by ``*`` is a (zero based)
       number identifying the :term:`locality`.

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the average
       overhead executing one |hpx|-thread phase (invocation) should be queried
       for. The worker thread number (given by the ``*`` is a (zero based)
       number identifying the worker thread. The number of available worker
       threads is usually specified on the command line for the application
       using the option :option:`--hpx:threads`. If no pool-name is specified
       the counter refers to the 'default' pool.
     * Returns the average time spent on overhead executing one |hpx|-thread
       phase (invocation) on the given :term:`locality` since application start.
       If the instance name is ``total`` the counter returns the average time
       spent on overhead while executing one |hpx|-thread phase (invocation) for
       all worker threads (cores) on that :term:`locality`. If the instance name
       is ``worker-thread#*`` the counter will return the average time spent on
       overhead executing one |hpx|-thread phase for all worker threads
       separately. This counter is available only if the configuration time
       constants ``HPX_WITH_THREAD_CUMULATIVE_COUNTS`` (default: ``ON``) and
       ``HPX_WITH_THREAD_IDLE_RATES`` are set to ``ON`` (default: ``OFF``). The
       unit of measure for this counter is nanosecond [ns].
     * None
   * * ``/threads/time/overall``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the overall
       time spent running the scheduler should be queried for. The
       :term:`locality` id (given by ``*`` is a (zero based) number identifying
       the :term:`locality`.

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the overall
       time spent running the scheduler should be queried for. The worker thread
       number (given by the ``*`` is a (zero based) number identifying the
       worker thread. The number of available worker threads is usually
       specified on the command line for the application using the option
       :option:`--hpx:threads`. If no pool-name is specified the counter refers
       to the 'default' pool.
     * Returns the overall time spent running the scheduler on the given
       :term:`locality` since application start. If the instance name is ``total``
       the counter returns the overall time spent running the scheduler for all
       worker threads (cores) on that :term:`locality`. If the instance name is
       ``worker-thread#*`` the counter will return the overall time spent running
       the scheduler for all worker threads separately. This counter is available
       only if the configuration time constant ``HPX_WITH_THREAD_IDLE_RATES`` is
       set to ``ON`` (default: ``OFF``). The unit of measure for this counter is
       nanosecond [ns].
     * None
   * * ``/threads/time/cumulative``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the overall
       time spent executing all |hpx|-threads should be queried for. The
       :term:`locality` id (given by ``*`` is a (zero based) number identifying
       the :term:`locality`.

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the overall
       time spent executing all |hpx|-threads should be queried for. The worker
       thread number (given by the ``*`` is a (zero based) number identifying
       the worker thread. The number of available worker threads is usually
       specified on the command line for the application using the option
       :option:`--hpx:threads`. If no pool-name is specified the counter refers
       to the 'default' pool.
     * Returns the overall time spent executing all |hpx|-threads on the given
       :term:`locality` since application start. If the instance name is ``total``
       the counter returns the overall time spent executing all |hpx|-threads for
       all worker threads (cores) on that :term:`locality`. If the instance name
       is ``worker-thread#*`` the counter will return the overall time spent
       executing all |hpx|-threads for all worker threads separately. This counter
       is available only if the configuration time constants
       ``HPX_THREAD_MAINTAIN_CUMULATIVE_COUNTS`` (default: ``ON``) and
       ``HPX_THREAD_MAINTAIN_IDLE_RATES`` are set to ``ON`` (default: ``OFF``).
     * None
   * * ``/threads/time/cumulative-overheads``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the overall
       overhead time incurred by executing all |hpx|-threads should be queried
       for. The :term:`locality` id (given by ``*`` is a (zero based) number
       identifying the :term:`locality`.

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the the
       overall overhead time incurred by executing all |hpx|-threads should be
       queried for. The worker thread number (given by the ``*`` is a (zero
       based) number identifying the worker thread. The number of available
       worker threads is usually specified on the command line for the
       application using the option :option:`--hpx:threads`. If no pool-name is
       specified the counter refers to the 'default' pool.
     * Returns the overall overhead time incurred executing all |hpx|-threads on
       the given :term:`locality` since application start. If the instance name
       is ``total`` the counter returns the overall overhead time incurred
       executing all |hpx|-threads for all worker threads (cores) on that
       :term:`locality`. If the instance name is ``worker-thread#*`` the counter
       will return the overall overhead time incurred executing all
       |hpx|-threads for all worker threads separately. This counter is
       available only if the configuration time constants
       ``HPX_THREAD_MAINTAIN_CUMULATIVE_COUNTS`` (default: ``ON``) and
       ``HPX_THREAD_MAINTAIN_IDLE_RATES`` are set to ``ON`` (default: ``OFF``).
       The unit of measure for this counter is nanosecond [ns].
     * None
   * * ``threads/count/instantaneous/<thread-state>``

       where:

       ``<thread-state>`` is one of the following: ``all``, ``active``,
       ``pending``, ``suspended``, ``terminated``, ``staged``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the current
       number of threads with the given state should be queried for. The
       :term:`locality` id (given by ``*`` is a (zero based) number identifying
       the :term:`locality`.

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the current
       number of threads with the given state should be queried for. The worker
       thread number (given by the ``*`` is a (zero based) number identifying
       the worker thread. The number of available worker threads is usually
       specified on the command line for the application using the option
       :option:`--hpx:threads`. If no pool-name is specified the counter refers
       to the 'default' pool.

       The ``staged`` thread state refers to registered tasks before they are
       converted to thread objects.
     * Returns the current number of |hpx|-threads having the given thread state
       on the given :term:`locality`. If the instance name is ``total`` the
       counter returns the current number of |hpx|-threads of the given state
       for all worker threads (cores) on that :term:`locality`. If the instance
       name is ``worker-thread#*`` the counter will return the current number of
       |hpx|-threads in the given state for all worker threads separately.
     * None
   * * ``threads/wait-time/<thread-state>``

       where:

       ``<thread-state>`` is one of the following: ``pending`` ``staged``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the average
       wait time of |hpx|-threads (pending) or thread descriptions (staged) with
       the given state should be queried for. The :term:`locality` id (given by
       ``*`` is a (zero based) number identifying the :term:`locality`.

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the average
       wait time for the given state should be queried for. The worker thread
       number (given by the ``*`` is a (zero based) number identifying the
       worker thread. The number of available worker threads is usually
       specified on the command line for the application using the option
       :option:`--hpx:threads`. If no pool-name is specified the counter refers
       to the 'default' pool.

       The ``staged`` thread state refers to the wait time of registered tasks
       before they are converted into thread objects, while the ``pending``
       thread state refers to the wait time of threads in any of the scheduling
       queues.
     * Returns the average wait time of |hpx|-threads (if the thread state is
       ``pending`` or of task descriptions (if the thread state is ``staged`` on
       the given :term:`locality` since application start. If the instance name
       is ``total`` the counter returns the wait time of |hpx|-threads of the
       given state for all worker threads (cores) on that :term:`locality`. If
       the instance name is ``worker-thread#*`` the counter will return the wait
       time of |hpx|-threads in the given state for all worker threads
       separately.

       These counters are available only if the compile time constant
       ``HPX_WITH_THREAD_QUEUE_WAITTIME`` was defined while compiling the |hpx|
       core library (default: ``OFF``). The unit of measure for this counter is
       nanosecond [ns].
     * None
   * * ``/threads/idle-rate``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the average
       idle rate of all (or one) worker threads should be queried for. The
       :term:`locality` id (given by ``*`` is a (zero based) number identifying
       the :term:`locality`

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the averaged
       idle rate should be queried for. The worker thread number (given by the
       ``*`` is a (zero based) number identifying the worker thread. The number
       of available worker threads is usually specified on the command line for
       the application using the option :option:`--hpx:threads`. If no pool-name
       is specified the counter refers to the 'default' pool.
     * Returns the average idle rate for the given worker thread(s) on the given
       :term:`locality`. The idle rate is defined as the ratio of the time spent
       on scheduling and management tasks and the overall time spent executing
       work since the application started. This counter is available only if the
       configuration time constant ``HPX_WITH_THREAD_IDLE_RATES`` is set to ``ON``
       (default: ``OFF``).
     * None
   * * ``/threads/creation-idle-rate``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the average
       creation idle rate of all (or one) worker threads should be queried for.
       The :term:`locality` id (given by ``*`` is a (zero based) number
       identifying the :term:`locality`.

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the averaged
       idle rate should be queried for. The worker thread number (given by the
       ``*`` is a (zero based) number identifying the worker thread. The number
       of available worker threads is usually specified on the command line for
       the application using the option :option:`--hpx:threads`. If no pool-name
       is specified the counter refers to the 'default' pool.
     * Returns the average idle rate for the given worker thread(s) on the given
       :term:`locality` which is caused by creating new threads. The creation idle
       rate is defined as the ratio of the time spent on creating new threads and
       the overall time spent executing work since the application started. This
       counter is available only if the configuration time constants
       ``HPX_WITH_THREAD_IDLE_RATES`` (default: ``OFF``) and
       ``HPX_WITH_THREAD_CREATION_AND_CLEANUP_RATES`` are set to ``ON``.
     * None
   * * ``/threads/cleanup-idle-rate``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or


       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the average
       cleanup idle rate of all (or one) worker threads should be queried for.
       The :term:`locality` id (given by ``*`` is a (zero based) number
       identifying the :term:`locality`.

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the averaged
       cleanup idle rate should be queried for. The worker thread number (given
       by the ``*`` is a (zero based) number identifying the worker thread. The
       number of available worker threads is usually specified on the command
       line for the application using the option :option:`--hpx:threads`. If no
       pool-name is specified the counter refers to the 'default' pool.
     * Returns the average idle rate for the given worker thread(s) on the given
       :term:`locality` which is caused by cleaning up terminated threads. The
       cleanup idle rate is defined as the ratio of the time spent on cleaning up
       terminated thread objects and the overall time spent executing work since
       the application started. This counter is available only if the
       configuration time constants ``HPX_WITH_THREAD_IDLE_RATES`` (default:
       ``OFF``) and ``HPX_WITH_THREAD_CREATION_AND_CLEANUP_RATES`` are set to
       ``ON``.
     * None
   * * ``/threadqueue/length``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the current
       length of all thread queues in the scheduler for all (or one) worker
       threads should be queried for. The :term:`locality` id (given by ``*`` is
       a (zero based) number identifying the :term:`locality`.

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the current
       length of all thread queues in the scheduler should be queried for. The
       worker thread number (given by the ``*`` is a (zero based) number
       identifying the worker thread. The number of available worker threads is
       usually specified on the command line for the application using the
       option :option:`--hpx:threads`. If no pool-name is specified the counter
       refers to the 'default' pool.
     * Returns the overall length of all queues for the given worker thread(s)
       on the given :term:`locality`.
     * None
   * * ``/threads/count/stack-unbinds``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the unbind
       (madvise) operations should be queried for. The :term:`locality` id is a
       (zero based) number identifying the :term:`locality`.
     * Returns the total number of |hpx|-thread unbind (madvise) operations
       performed for the referenced :term:`locality`. Note that this counter is
       not available on Windows based platforms.
     * None
   * * ``/threads/count/stack-recycles``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the recycling
       operations should be queried for. The :term:`locality` id is a (zero
       based) number identifying the :term:`locality`.
     * Returns the total number of |hpx|-thread recycling operations performed.
     * None
   * * ``/threads/count/stolen-from-pending``
     * ``locality#*/total``

          where:

       ``*`` is the :term:`locality` id of the :term:`locality` the number of
       'stole' threads should be queried for. The :term:`locality` id is a (zero
       based) number identifying the :term:`locality`.
     * Returns the total number of |hpx|-threads 'stolen' from the pending
       thread queue by a neighboring thread worker thread (these threads are
       executed by a different worker thread than they were initially scheduled
       on). This counter is available only if the configuration time constant
       ``HPX_WITH_THREAD_STEALING_COUNTS`` is set to ``ON`` (default: ``ON``).
     * None
   * * ``/threads/count/pending-misses``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the number of
       pending queue misses of all (or one) worker threads should be queried
       for. The :term:`locality` id (given by ``*`` is a (zero based) number
       identifying the :term:`locality`

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the number of
       pending queue misses should be queried for. The worker thread number
       (given by the ``*`` is a (zero based) number identifying the worker
       thread. The number of available worker threads is usually specified on
       the command line for the application using the option
       :option:`--hpx:threads`. If no pool-name is specified the counter refers
       to the 'default' pool.
     * Returns the total number of times that the referenced worker-thread on
       the referenced :term:`locality` failed to find pending |hpx|-threads in
       its associated queue. This counter is available only if the configuration
       time constant ``HPX_WITH_THREAD_STEALING_COUNTS`` is set to ``ON``
       (default: ``ON``).
     * None
   * * ``/threads/count/pending-accesses``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the number of
       pending queue accesses of all (or one) worker threads should be queried
       for. The :term:`locality` id (given by ``*`` is a (zero based) number
       identifying the :term:`locality`

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the number of
       pending queue accesses should be queried for. The worker thread number
       (given by the ``*`` is a (zero based) number identifying the worker
       thread. The number of available worker threads is usually specified on
       the command line for the application using the option
       :option:`--hpx:threads`. If no pool-name is specified the counter refers
       to the 'default' pool.
     * Returns the total number of times that the referenced worker-thread on
       the referenced :term:`locality` looked for pending |hpx|-threads in its
       associated queue. This counter is available only if the configuration
       time constant ``HPX_WITH_THREAD_STEALING_COUNTS`` is set to ``ON``
       (default: ``ON``).
     * None
   * * ``/threads/count/stolen-from-staged``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the number of
       |hpx|-threads stolen from the staged queue of all (or one) worker threads
       should be queried for. The :term:`locality` id (given by ``*`` is a (zero
       based) number identifying the :term:`locality`.

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the number of
       |hpx|-threads stolen from the staged queue should be queried for. The
       worker thread number (given by the ``*`` is a (zero based) number
       identifying the worker thread. The number of available worker threads is
       usually specified on the command line for the application using the
       option :option:`--hpx:threads`. If no pool-name is specified the counter
       refers to the 'default' pool.
     * Returns the total number of |hpx|-threads 'stolen' from the staged thread
       queue by a neighboring worker thread (these threads are executed by a
       different worker thread than they were initially scheduled on). This
       counter is available only if the configuration time constant
       ``HPX_WITH_THREAD_STEALING_COUNTS`` is set to ``ON`` (default: ``ON``).
     * None
   * * ``/threads/count/stolen-to-pending``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the number of
       |hpx|-threads stolen to the pending queue of all (or one) worker threads
       should be queried for. The :term:`locality` id (given by ``*`` is a (zero
       based) number identifying the :term:`locality`.

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the number of
       |hpx|-threads stolen to the pending queue should be queried for. The
       worker thread number (given by the ``*`` is a (zero based) number
       identifying the worker thread. The number of available worker threads is
       usually specified on the command line for the application using the
       option :option:`--hpx:threads`. If no pool-name is specified the counter
       refers to the 'default' pool.
     * Returns the total number of |hpx|-threads 'stolen' to the pending thread
       queue of the worker thread (these threads are executed by a different
       worker thread than they were initially scheduled on). This counter is
       available only if the configuration time constant
       ``HPX_WITH_THREAD_STEALING_COUNTS`` is set to ``ON`` (default: ``ON``).
     * None
   * * ``/threads/count/stolen-to-staged``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the number of
       |hpx|-threads stolen to the staged queue of all (or one) worker threads
       should be queried for. The :term:`locality` id (given by ``*`` is a (zero
       based) number identifying the :term:`locality`.

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the number of
       |hpx|-threads stolen to the staged queue should be queried for. The
       worker thread number (given by the ``*`` is a (zero based) worker thread
       number (given by the ``*`` is a (zero based) number identifying the
       worker thread. The number of available worker threads is usually
       specified on the command line for the application using the option
       :option:`--hpx:threads`. If no pool-name is specified the counter refers
       to the 'default' pool.
     * Returns the total number of |hpx|-threads 'stolen' to the staged thread
       queue of a neighboring worker thread (these threads are executed by a
       different worker thread than they were initially scheduled on). This
       counter is available only if the configuration time constant
       ``HPX_WITH_THREAD_STEALING_COUNTS`` is set to ``ON`` (default: ``ON``).
     * None
   * * ``/threads/count/objects``
     * ``locality#*/total`` or

       ``locality#*/allocator#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the current
       (cumulative) number of all created |hpx|-thread objects should be queried
       for. The :term:`locality` id (given by ``*`` is a (zero based) number
       identifying the :term:`locality`.

       ``allocator#*`` is defining the number of the allocator instance using
       which the threads have been created. |hpx| uses a varying number of
       allocators to create (and recycle) |hpx|-thread objects, most likely
       these counters are of use for debugging purposes only. The allocator id
       (given by ``*`` is a (zero based) number identifying the allocator to
       query.
     * Returns the total number of |hpx|-thread objects created. Note that
       thread objects are reused to improve system performance, thus this number
       does not reflect the number of actually executed (retired) |hpx|-threads.
     * None
   * * ``/scheduler/utilization/instantaneous``
     * ``locality#*/total``

       where:

       ``locality#*`` is defining the :term:`locality` for which the current
       (instantaneous) scheduler utilization queried for. The :term:`locality`
       id (given by ``*`` is a (zero based) number identifying the
       :term:`locality`.
     * Returns the total (instantaneous) scheduler utilization. This is the
        current percentage of scheduler threads executing |hpx| threads.
     * Percent
   * * ``/threads/idle-loop-count/instantaneous``
     * ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the current
       current accumulated value of all idle-loop counters of all worker threads
       should be queried. The :term:`locality` id (given by ``*`` is a (zero
       based) number identifying the :term:`locality`.

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the current
       value of the idle-loop counter should be queried for. The worker thread
       number (given by the ``*`` is a (zero based) worker thread number (given
       by the ``*`` is a (zero based) number identifying the worker thread. The
       number of available worker threads is usually specified on the command
       line for the application using the option :option:`--hpx:threads`. If no
       pool-name is specified the counter refers to the 'default' pool.
     * Returns the current (instantaneous) idle-loop count for the given
       |hpx|- worker thread or the accumulated value for all worker threads.
     * None
   * * ``/threads/busy-loop-count/instantaneous``
     * ``locality#*/worker-thread#*`` or

       ``locality#*/pool#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the current
       current accumulated value of all busy-loop counters of all worker threads
       should be queried. The :term:`locality` id (given by ``*`` is a (zero
       based) number identifying the :term:`locality`.

       ``pool#*`` is defining the pool for which the current value of the
       idle-loop counter should be queried for.

       ``worker-thread#*`` is defining the worker thread for which the current
       value of the busy-loop counter should be queried for. The worker thread
       number (given by the ``*`` is a (zero based) worker thread number (given
       by the ``*`` is a (zero based) number identifying the worker thread. The
       number of available worker threads is usually specified on the command
       line for the application using the option :option:`--hpx:threads`. If no
       pool-name is specified the counter refers to the 'default' pool.
     * Returns the current (instantaneous) busy-loop count for the given |hpx|-
       worker thread or the accumulated value for all worker threads.
     * None
   * * ``/threads/time/background-work-duration``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*``

       where:

       ``locality#*`` is defining the locality for which the overall time spent
       performing background work should be queried for. The locality id (given
       by ``*``) is a (zero based) number identifying the locality.

       ``worker-thread#*`` is defining the worker thread for which the overall
       time spent performing background work should be queried for. The worker
       thread number (given by the ``*``) is a (zero based) number identifying
       the worker thread. The number of available worker threads is usually
       specified on the command line for the application using the option
       :option:`--hpx:threads`.

     * Returns the overall time spent performing background work on the given
       locality since application start. If the instance name is ``total`` the
       counter returns the overall time spent performing background work for all
       worker threads (cores) on that locality. If the instance name is
       ``worker-thread#*`` the counter will return the overall time spent
       performing background work for all worker threads separately. This
       counter is available only if the configuration time constants
       ``HPX_WITH_BACKGROUND_THREAD_COUNTERS`` (default: ``OFF``) and
       ``HPX_WITH_THREAD_IDLE_RATES`` are set to ``ON`` (default: ``OFF``). The
       unit of measure for this counter is nanosecond [ns].

     * None
   * * ``/threads/background-overhead``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*``

       where:

       ``locality#*`` is defining the locality for which the background overhead
       should be queried for. The locality id (given by ``*``) is a (zero based)
       number identifying the locality.

       ``worker-thread#*`` is defining the worker thread for which the
       background overhead should be queried for. The worker thread number
       (given by the ``*``) is a (zero based) number identifying the worker
       thread. The number of available worker threads is usually specified on
       the command line for the application using the option
       :option:`--hpx:threads`.
     * Returns the background overhead on the given locality since application
       start. If the instance name is ``total`` the counter returns the
       background overhead for all worker threads (cores) on that locality. If
       the instance name is ``worker-thread#*`` the counter will return
       background overhead for all worker threads separately. This counter is
       available only if the configuration time constants
       ``HPX_WITH_BACKGROUND_THREAD_COUNTERS`` (default: ``OFF``) and
       ``HPX_WITH_THREAD_IDLE_RATES`` are set to ``ON`` (default: ``OFF``). The
       unit of measure displayed for this counter is 0.1%.
     * None
   * * ``/threads/time/background-send-duration``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*``

       where:

       ``locality#*`` is defining the locality for which the overall time spent
       performing background work related to sending parcels should be queried
       for. The locality id (given by ``*``) is a (zero based) number
       identifying the locality.

       ``worker-thread#*`` is defining the worker thread for which the overall
       time spent performing background work related to sending parcels should
       be queried for. The worker thread number (given by the ``*``) is a (zero
       based) number identifying the worker thread. The number of available
       worker threads is usually specified on the command line for the
       application using the option :option:`--hpx:threads`.

     * Returns the overall time spent performing background work related to
       sending parcels on the given locality since application start. If the
       instance name is ``total`` the counter returns the overall time spent
       performing background work for all worker threads (cores) on that
       locality. If the instance name is ``worker-thread#*`` the counter will
       return the overall time spent performing background work for all worker
       threads separately. This counter is available only if the configuration
       time constants ``HPX_WITH_BACKGROUND_THREAD_COUNTERS`` (default: ``OFF``)
       and ``HPX_WITH_THREAD_IDLE_RATES`` are set to ``ON`` (default: ``OFF``).
       The unit of measure for this counter is nanosecond [ns].

       This counter will currently return meaningful values for the MPI
       parcelport only.

     * None
   * * ``/threads/background-send-overhead``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*``

       where:

       ``locality#*`` is defining the locality for which the background overhead
       related to sending parcels should be queried for. The locality id (given
       by ``*``) is a (zero based) number identifying the locality.

       ``worker-thread#*`` is defining the worker thread for which the
       background overhead related to sending parcels should be queried for.
       The worker thread number (given by the ``*``) is a (zero based) number
       identifying the worker thread. The number of available worker threads is
       usually specified on the command line for the application using the option
       :option:`--hpx:threads`.
     * Returns the background overhead related to sending parcels on the given
       locality since application start. If the instance name is ``total`` the
       counter returns the background overhead for all worker threads (cores) on
       that locality. If the instance name is ``worker-thread#*`` the counter
       will return background overhead for all worker threads separately. This
       counter is available only if the configuration time constants
       ``HPX_WITH_BACKGROUND_THREAD_COUNTERS`` (default: ``OFF``) and
       ``HPX_WITH_THREAD_IDLE_RATES`` are set to ``ON`` (default: ``OFF``). The
       unit of measure displayed for this counter is 0.1%.

       This counter will currently return meaningful values for the MPI
       parcelport only.
     * None
   * * ``/threads/time/background-receive-duration``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*``

       where:

       ``locality#*`` is defining the locality for which the overall time spent
       performing background work related to receiving parcels should be queried
       for. The locality id (given by ``*``) is a (zero based) number identifying
       the locality.

       ``worker-thread#*`` is defining the worker thread for which the overall
       time spent performing background work related to receiving parcels should
       be queried for. The worker thread number (given by the ``*``) is a (zero
       based) number identifying the worker thread. The number of available
       worker threads is usually specified on the command line for the
       application using the option :option:`--hpx:threads`.

     * Returns the overall time spent performing background work related to
       receiving parcels on the given locality since application start. If the
       instance name is ``total`` the counter returns the overall time spent
       performing background work for all worker threads (cores) on that
       locality. If the instance name is ``worker-thread#*`` the counter will
       return the overall time spent performing background work for all worker
       threads separately. This counter is available only if the configuration
       time constants ``HPX_WITH_BACKGROUND_THREAD_COUNTERS`` (default: ``OFF``)
       and ``HPX_WITH_THREAD_IDLE_RATES`` are set to ``ON`` (default: ``OFF``).
       The unit of measure for this counter is nanosecond [ns].

       This counter will currently return meaningful values for the MPI
       parcelport only.
     * None
   * * ``/threads/background-receive-overhead``
     * ``locality#*/total`` or

       ``locality#*/worker-thread#*``

       where:

       ``locality#*`` is defining the locality for which the background overhead
       related to receiving should be queried for. The locality id (given by
       ``*``) is a (zero based) number identifying the locality.

       ``worker-thread#*`` is defining the worker thread for which the
       background overhead related to receiving parcels should be queried for.
       The worker thread number (given by the ``*``) is a (zero based) number
       identifying the worker thread. The number of available worker threads is
       usually specified on the command line for the application using the option
       :option:`--hpx:threads`.
     * Returns the background overhead related to receiving parcels on the given
       locality since application start. If the instance name is ``total`` the
       counter returns the background overhead for all worker threads (cores) on
       that locality. If the instance name is ``worker-thread#*`` the counter
       will return background overhead for all worker threads separately. This
       counter is available only if the configuration time constants
       ``HPX_WITH_BACKGROUND_THREAD_COUNTERS`` (default: ``OFF``) and
       ``HPX_WITH_THREAD_IDLE_RATES`` are set to ``ON`` (default: ``OFF``). The
       unit of measure displayed for this counter is 0.1%.

       This counter will currently return meaningful values for the MPI
       parcelport only.
     * None

.. list-table:: General performance counters exposing characteristics of localities

   * * Counter type
     * Counter instance formatting
     * Description
     * Parameters
   * * ``/runtime/count/component``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the number of
       components should be queried. The :term:`locality` id is a (zero based)
       number identifying the :term:`locality`.
     * Returns the overall number of currently active components of the
       specified type on the given :term:`locality`.
     * The type of the component. This is the string which has been used while
       registering the component with |hpx|, e.g. which has been passed as the
       second parameter to the macro :c:macro:`HPX_REGISTER_COMPONENT`.
   * * ``/runtime/count/action-invocation``
     * ``locality#*/total``

         where:

       ``*`` is the :term:`locality` id of the locality the number of action
       invocations should be queried. The :term:`locality` id is a (zero based)
       number identifying the :term:`locality`.
     * Returns the overall (local) invocation count of the specified action type
       on the given :term:`locality`.
     * The action type. This is the string which has been used while registering
       the action with |hpx|, e.g. which has been passed as the second parameter
       to the macro :c:macro:`HPX_REGISTER_ACTION` or
       :c:macro:`HPX_REGISTER_ACTION_ID`.
   * * ``/runtime/count/remote-action-invocation``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the number of
       action invocations should be queried. The :term:`locality` id is a (zero
       based) number identifying the :term:`locality`.
     * Returns the overall (remote) invocation count of the specified action
       type on the given :term:`locality`.
     * The action type. This is the string which has been used while registering
       the action with |hpx|, e.g. which has been passed as the second parameter
       to the macro :c:macro:`HPX_REGISTER_ACTION` or
       :c:macro:`HPX_REGISTER_ACTION_ID`.
   * * ``/runtime/uptime``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the system
       uptime should be queried. The :term:`locality` id is a (zero based)
       number identifying the :term:`locality`.
     * Returns the overall time since application start on the given
       :term:`locality` in nanoseconds.
     * None
   * * ``/runtime/memory/virtual``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the allocated
       virtual memory should be queried. The :term:`locality` id is a (zero
       based) number identifying the :term:`locality`.
     * Returns the amount of virtual memory currently allocated by the
       referenced :term:`locality` (in bytes).
     * None
   * * ``/runtime/memory/resident``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the allocated
       resident memory should be queried. The :term:`locality` id is a (zero
       based) number identifying the :term:`locality`.
     * Returns the amount of resident memory currently allocated by the
       referenced :term:`locality` (in bytes).
     * None
   * * ``/runtime/memory/total``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the total
       available memory should be queried. The :term:`locality` id is a (zero
       based) number identifying the :term:`locality`. Note: only supported in
       Linux.
     * Returns the total available memory for use by the referenced
        :term:`locality` (in bytes). This counter is available on Linux and
        Windows systems only.
     * None
   * * ``/runtime/io/read_bytes_issued``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the number of
       bytes read should be queried. The :term:`locality` id is a (zero based)
       number identifying the :term:`locality`.
     * Returns the number of bytes read by the process (aggregate of count
       arguments passed to read() call or its analogues). This performance
       counter is available only on systems which expose the related data
       through the /proc file system.
     * None
   * * ``/runtime/io/write_bytes_issued``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the number of
       bytes written should be queried. The :term:`locality` id is a (zero
       based) number identifying the :term:`locality`.
     * Returns the number of bytes written by the process (aggregate of count
       arguments passed to write() call or its analogues). This performance
       counter is available only on systems which expose the related data
       through the /proc file system.
     * None
   * * ``/runtime/io/read_syscalls``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the number of
       system calls should be queried. The :term:`locality` id is a (zero based)
       number identifying the :term:`locality`.
     * Returns the number of system calls that perform I/O reads. This
       performance counter is available only on systems which expose the
       related data through the /proc file system.
     * None
   * * ``/runtime/io/write_syscalls``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the number of
       system calls should be queried. The :term:`locality` id is a (zero based)
       number identifying the :term:`locality`.
     * Returns the number of system calls that perform I/O writes. This
       performance counter is available only on systems which expose the
       related data through the /proc file system.
     * None
   * * ``/runtime/io/read_bytes_transferred``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the number of
       bytes transferred should be queried. The :term:`locality` id is a (zero
       based) number identifying the :term:`locality`.
     * Returns the number of bytes retrieved from storage by I/O operations.
       This performance counter is available only on systems which expose the
       related data through the /proc file system.
     * None
   * * ``/runtime/io/write_bytes_transferred``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the number of
       bytes transferred should be queried. The :term:`locality` id is a (zero
       based) number identifying the :term:`locality`.
     * Returns the number of bytes retrieved from storage by I/O operations.
       This performance counter is available only on systems which expose the
       related data through the /proc file system.
     * None
   * * ``/runtime/io/write_bytes_cancelled``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the number of
       bytes not being transferred should be queried. The :term:`locality` id is
       a (zero based) number identifying the :term:`locality`.
     * Returns the number of bytes accounted by write_bytes_transferred that
       has not been ultimately stored due to truncation or deletion. This
       performance counter is available only on systems which expose the
       related data through the /proc file system.
     * None

.. list-table:: Performance counters exposing PAPI hardware counters

   * * Counter type
     * Counter instance formatting
     * Description
     * Parameters
   * * ``/papi/<papi_event>``

       where:

       ``<papi_event>`` is the name of the PAPI event to expose as a performance
       counter (such as ``PAPI_SR_INS``). Note that the list of available PAPI
       events changes depending on the used architecture.

       For a full list of available PAPI events and their (short) description
       use the ``--hpx:list-counters`` and ``--hpx:papi-event-info=all`` command
       line options.

     * ``locality#*/total`` or

       ``locality#*/worker-thread#*``

       where:

       ``locality#*`` is defining the :term:`locality` for which the current
       current accumulated value of all busy-loop counters of all worker threads
       should be queried. The :term:`locality` id (given by ``*``) is a (zero
       based) number identifying the :term:`locality`.

       ``worker-thread#*`` is defining the worker thread for which the current
       value of the busy-loop counter should be queried for. The worker thread
       number (given by the ``*``) is a (zero based) worker thread number (given
       by the ``*``) is a (zero based) number identifying the worker thread. The
       number of available worker threads is usually specified on the command
       line for the application using the option :option:`--hpx:threads`.

     * This counter returns the current count of occurrences of the specified
       PAPI event. This counter is available only if the configuration time
       constant ``HPX_WITH_PAPI`` is set to ``ON`` (default: ``OFF``).
     * None

.. list-table:: Performance counters for general statistics

   * * Counter type
     * Counter instance formatting
     * Description
     * Parameters
   * * ``/statistics/average``
     * Any full performance counter name. The referenced performance counter is
       queried at fixed time intervals as specified by the first parameter.
     * Returns the current average (mean) value calculated based on the values
       queried from the underlying counter (the one specified as the instance
       name).
     * Any parameter will be interpreted as a list of up to two comma separated
       (integer) values, where the first is the time interval (in milliseconds)
       at which the underlying counter should be queried. If no value is
       specified, the counter will assume ``1000`` [ms] as the default. The
       second value can be either ``0`` or ``1`` and specifies whether the
       underlying counter should be reset during evaluation ``1`` or not ``0``.
       The default value is ``0``.

   * * ``/statistics/rolling_average``
     * Any full performance counter name. The referenced performance counter is
       queried at fixed time intervals as specified by the first parameter.
     * Returns the current rolling average (mean) value calculated based on the
       values queried from the underlying counter (the one specified as the
       instance name).
     * Any parameter will be interpreted as a list of up to three comma
       separated (integer) values, where the first is the time interval (in
       milliseconds) at which the underlying counter should be queried. If no
       value is specified, the counter will assume ``1000`` [ms] as the default.
       The second value will be interpreted as the size of the rolling window
       (the number of latest values to use to calculate the rolling average).
       The default value for this is ``10``. The third value can be
       either ``0`` or ``1`` and specifies whether the underlying counter should
       be reset during evaluation ``1`` or not ``0``. The default value is ``0``.

   * * ``/statistics/stddev``
     * Any full performance counter name. The referenced performance counter is
       queried at fixed time intervals as specified by the first parameter.
     * Returns the current standard deviation (stddev) value calculated based on
       the values queried from the underlying counter (the one specified as the
       instance name).
     * Any parameter will be interpreted as a list of up to two comma separated
       (integer) values, where the first is the time interval (in milliseconds)
       at which the underlying counter should be queried. If no value is
       specified, the counter will assume ``1000`` [ms] as the default. The
       second value can be either ``0`` or ``1`` and specifies whether the
       underlying counter should be reset during evaluation ``1`` or not ``0``.
       The default value is ``0``.

   * * ``/statistics/rolling_stddev``
     * Any full performance counter name. The referenced performance counter is
       queried at fixed time intervals as specified by the first parameter.
     * Returns the current rolling variance (stddev) value calculated based on
       the values queried from the underlying counter (the one specified as the
       instance name).
     * Any parameter will be interpreted as a list of up to three comma
       separated (integer) values, where the first is the time interval (in
       milliseconds) at which the underlying counter should be queried. If no
       value is specified, the counter will assume ``1000`` [ms] as the default.
       The second value will be interpreted as the size of the rolling window
       (the number of latest values to use to calculate the rolling average).
       The default value for this is ``10``. The third value can be either ``0``
       or ``1`` and specifies whether the underlying counter should be reset
       during evaluation ``1`` or not ``0``. The default value is ``0``.

   * * ``/statistics/median``
     * Any full performance counter name. The referenced performance counter is
       queried at fixed time intervals as specified by the first parameter.
     * Returns the current (statistically estimated) median value calculated
       based on the values queried from the underlying counter (the one
       specified as the instance name).
     * Any parameter will be interpreted as a list of up to two comma separated
       (integer) values, where the first is the time interval (in milliseconds)
       at which the underlying counter should be queried. If no value is
       specified, the counter will assume ``1000`` [ms] as the default. The
       second value can be either ``0`` or ``1`` and specifies whether the
       underlying counter should be reset during evaluation ``1`` or not ``0``.
       The default value is ``0``.

   * * ``/statistics/max``
     * Any full performance counter name. The referenced performance counter is
       queried at fixed time intervals as specified by the first parameter.
     * Returns the current maximum value calculated based on the values queried
       from the underlying counter (the one specified as the instance name).
     * Any parameter will be interpreted as a list of up to two comma separated
       (integer) values, where the first is the time interval (in milliseconds)
       at which the underlying counter should be queried. If no value is
       specified, the counter will assume ``1000`` [ms] as the default. The
       second value can be either ``0`` or ``1`` and specifies whether the
       underlying counter should be reset during evaluation ``1`` or not ``0``.
       The default value is ``0``.

   * * ``/statistics/rolling_max``
     * Any full performance counter name. The referenced performance counter is
       queried at fixed time intervals as specified by the first parameter.
     * Returns the current rolling maximum value calculated based on the values
       queried from the underlying counter (the one specified as the instance
       name).
     * Any parameter will be interpreted as a list of up to three comma
       separated (integer) values, where the first is the time interval (in
       milliseconds) at which the underlying counter should be queried. If no
       value is specified, the counter will assume ``1000`` [ms] as the default.
       The second value will be interpreted as the size of the rolling window
       (the number of latest values to use to calculate the rolling average).
       The default value for this is ``10``. The third value can be either ``0``
       or ``1`` and specifies whether the underlying counter should be reset
       during evaluation ``1`` or not ``0``. The default value is ``0``.

   * * ``/statistics/min``
     * Any full performance counter name. The referenced performance counter is
       queried at fixed time intervals as specified by the first parameter.
     * Returns the current minimum value calculated based on the values queried
       from the underlying counter (the one specified as the instance name).
     * Any parameter will be interpreted as a list of up to two comma separated
       (integer) values, where the first is the time interval (in milliseconds)
       at which the underlying counter should be queried. If no value is
       specified, the counter will assume ``1000`` [ms] as the default. The
       second value can be either ``0`` or ``1`` and specifies whether the
       underlying counter should be reset during evaluation ``1`` or not ``0``.
       The default value is ``0``.

   * * ``/statistics/rolling_min``
     * Any full performance counter name. The referenced performance counter is
       queried at fixed time intervals as specified by the first parameter.
     * Returns the current rolling minimum value calculated based on the values
       queried from the underlying counter (the one specified as the instance
       name).
     * Any parameter will be interpreted as a list of up to three comma
       separated (integer) values, where the first is the time interval (in
       milliseconds) at which the underlying counter should be queried. If no
       value is specified, the counter will assume ``1000`` [ms] as the default.
       The second value will be interpreted as the size of the rolling window
       (the number of latest values to use to calculate the rolling average).
       The default value for this is ``10``. The third value can be either ``0``
       or ``1`` and specifies whether the underlying counter should be reset
       during evaluation ``1`` or not ``0``. The default value is ``0``.

.. list-table:: Performance counters for elementary arithmetic operations

   * * Counter type
     * Counter instance formatting
     * Description
     * Parameters
   * * ``/arithmetics/add``
     * None
     * Returns the sum calculated based on the values queried from the
       underlying counters (the ones specified as the parameters).
     * The parameter will be interpreted as a comma separated list of full
       performance counter names which are queried whenever this counter is
       accessed. Any wildcards in the counter names will be expanded.
   * * ``/arithmetics/subtract``
     * None
     * Returns the difference calculated based on the values queried from the
       underlying counters (the ones specified as the parameters).
     * The parameter will be interpreted as a comma separated list of full
       performance counter names which are queried whenever this counter is
       accessed. Any wildcards in the counter names will be expanded.
   * * ``/arithmetics/multiply``
     * None
     * Returns the product calculated based on the values queried from the
       underlying counters (the ones specified as the parameters).
     * The parameter will be interpreted as a comma separated list of full
       performance counter names which are queried whenever this counter is
       accessed. Any wildcards in the counter names will be expanded.
   * * ``/arithmetics/divide``
     * None
     * Returns the result of division of the values queried from the
       underlying counters (the ones specified as the parameters).
     * The parameter will be interpreted as a comma separated list of full
       performance counter names which are queried whenever this counter is
       accessed. Any wildcards in the counter names will be expanded.
   * * ``/arithmetics/mean``
     * None
     * Returns the average value of all values queried from the
       underlying counters (the ones specified as the parameters).
     * The parameter will be interpreted as a comma separated list of full
       performance counter names which are queried whenever this counter is
       accessed. Any wildcards in the counter names will be expanded.
   * * ``/arithmetics/variance``
     * None
     * Returns the standard deviation of all values queried from the underlying
       counters (the ones specified as the parameters).
     * The parameter will be interpreted as a comma separated list of full
       performance counter names which are queried whenever this counter is
       accessed. Any wildcards in the counter names will be expanded.
   * * ``/arithmetics/median``
     * None
     * Returns the median value of all values queried from the underlying
       counters (the ones specified as the parameters).
     * The parameter will be interpreted as a comma separated list of full
       performance counter names which are queried whenever this counter is
       accessed. Any wildcards in the counter names will be expanded.
   * * ``/arithmetics/min``
     * None
     * Returns the minimum value of all values queried from the underlying
       counters (the ones specified as the parameters).
     * The parameter will be interpreted as a comma separated list of full
       performance counter names which are queried whenever this counter is
       accessed. Any wildcards in the counter names will be expanded.
   * * ``/arithmetics/max``
     * None
     * Returns the maximum value of all values queried from the
       underlying counters (the ones specified as the parameters).
     * The parameter will be interpreted as a comma separated list of full
       performance counter names which are queried whenever this counter is
       accessed. Any wildcards in the counter names will be expanded.
   * * ``/arithmetics/count``
     * None
     * Returns the count value of all values queried from the underlying
       counters (the ones specified as the parameters).
     * The parameter will be interpreted as a comma separated list of full
       performance counter names which are queried whenever this counter is
       accessed. Any wildcards in the counter names will be expanded.

.. note::

   The ``/arithmetics`` counters can consume an arbitrary number of other
   counters. For this reason those have to be specified as parameters (a comma
   separated list of counters appended after a ``'@'``). For instance:

   .. code-block:: bash

      ./bin/hello_world_distributed -t2 \
          --hpx:print-counter=/threads{locality#0/worker-thread#*}/count/cumulative \
          --hpx:print-counter=/arithmetics/add@/threads{locality#0/worker-thread#*}/count/cumulative
      hello world from OS-thread 0 on locality 0
      hello world from OS-thread 1 on locality 0
      /threads{locality#0/worker-thread#0}/count/cumulative,1,0.515640,[s],25
      /threads{locality#0/worker-thread#1}/count/cumulative,1,0.515520,[s],36
      /arithmetics/add@/threads{locality#0/worker-thread#*}/count/cumulative,1,0.516445,[s],64

   Since all wildcards in the parameters are expanded, this example is fully
   equivalent to specifying both counters separately to ``/arithmetics/add``:

   .. code-block:: bash

      ./bin/hello_world_distributed -t2 \
          --hpx:print-counter=/threads{locality#0/worker-thread#*}/count/cumulative \
          --hpx:print-counter=/arithmetics/add@\
              /threads{locality#0/worker-thread#0}/count/cumulative,\
              /threads{locality#0/worker-thread#1}/count/cumulative

.. list-table:: Performance counters tracking :term:`parcel` coalescing

   * * Counter type
     * Counter instance formatting
     * Description
     * Parameters

   * * ``/coalescing/count/parcels``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the number of
       parcels for the given action should be queried for. The :term:`locality`
       id is a (zero based) number identifying the :term:`locality`.
     * Returns the number of parcels handled by the message handler
       associated with the action which is given by the counter parameter.
     * The action type. This is the string which has been used while registering
       the action with |hpx|, e.g. which has been passed as the second parameter
       to the macro :c:macro:`HPX_REGISTER_ACTION` or
       :c:macro:`HPX_REGISTER_ACTION_ID`.

   * * ``/coalescing/count/messages``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the number of
       messages for the given action should be queried for. The :term:`locality`
       id is a (zero based) number identifying the :term:`locality`.
     * Returns the number of messages generated by the message handler
       associated with the action which is given by the counter parameter.
     * The action type. This is the string which has been used while registering
       the action with |hpx|, e.g. which has been passed as the second parameter
       to the macro :c:macro:`HPX_REGISTER_ACTION` or
       :c:macro:`HPX_REGISTER_ACTION_ID`.

   * * ``/coalescing/count/average-parcels-per-message``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the number of
       messages for the given action should be queried for. The :term:`locality`
       id is a (zero based) number identifying the :term:`locality`.
     * Returns the average number of parcels sent in a message generated by the
       message handler associated with the action which is given by the counter
       parameter.
     * The action type. This is the string which has been used while registering
       the action with |hpx|, e.g. which has been passed as the second parameter
       to the macro :c:macro:`HPX_REGISTER_ACTION` or
       :c:macro:`HPX_REGISTER_ACTION_ID`

   * * ``/coalescing/time/average-parcel-arrival``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the average time
       between parcels for the given action should be queried for. The
       :term:`locality` id is a (zero based) number identifying the
       :term:`locality`.
     * Returns the average time between arriving parcels for the
       action which is given by the counter parameter.
     * The action type. This is the string which has been used while registering
       the action with |hpx|, e.g. which has been passed as the second parameter
       to the macro :c:macro:`HPX_REGISTER_ACTION` or
       :c:macro:`HPX_REGISTER_ACTION_ID`

   * * ``/coalescing/time/parcel-arrival-histogram``
     * ``locality#*/total``

       where:

       ``*`` is the :term:`locality` id of the :term:`locality` the average time
       between parcels for the given action should be queried for. The
       :term:`locality` id is a (zero based) number identifying the
       :term:`locality`.
     * Returns a histogram representing the times between arriving parcels for
       the action which is given by the counter parameter.

       This counter returns an array of values, where the first three values
       represent the three parameters used for the histogram followed by one
       value for each of the histogram buckets.

       The first unit of measure displayed for this counter ``[ns]`` refers to
       the lower and upper boundary values in the returned histogram data only.
       The second unit of measure displayed ``[0.1%]`` refers to the actual
       histogram data.

       For each bucket the counter shows a value between ``0`` and ``1000``
       which corresponds to a percentage value between ``0%`` and ``100%``.

     * The action type and optional histogram parameters. The action type is
       the string which has been used while registering the action with |hpx|,
       e.g. which has been passed as the second parameter to the macro
       :c:macro:`HPX_REGISTER_ACTION` or :c:macro:`HPX_REGISTER_ACTION_ID`.

       The action type may be followed by a comma separated list of up-to three
       numbers: the lower and upper boundaries for the collected histogram, and
       the number of buckets for the histogram to generate. By default these
       three numbers will be assumed to be ``0`` (``[ns]``, lower
       bound), ``1000000`` (``[ns]``, upper bound), and ``20`` (number of
       buckets to generate).

.. note::

   The performance counters related to :term:`parcel` coalescing are available only if
   the configuration time constant ``HPX_WITH_PARCEL_COALESCING`` is set to
   ``ON`` (default: ``ON``). However, even in this case it will be available
   only for actions that are enabled for parcel coalescing (see the
   macros :c:macro:`HPX_ACTION_USES_MESSAGE_COALESCING` and
   :c:macro:`HPX_ACTION_USES_MESSAGE_COALESCING_NOTHROW`).

.. [#] A message can potentially consist of more than one :term:`parcel`.

APEX integration
================

|hpx| provides integration with |apex|_, which is a framework for application
profiling using task timers and various performance counters. It can be added as
a ``git`` submodule by turning on the option :option:`HPX_WITH_APEX:BOOL` during
|cmake| configuration. |tau|_ is an optional dependency when using |apex|.

To build |hpx| with |apex|, add :option:`HPX_WITH_APEX`\ ``=ON``, and,
optionally, ``TAU_ROOT=$PATH_TO_TAU`` to your |cmake| configuration. In
addition, you can override the tag used for |apex| with the
:option:`HPX_WITH_APEX_TAG` option. Please see the |apex_hpx_doc|_ for detailed
instructions on using |apex| with |hpx|.
