..
    Copyright (c) 2014 Adrian Serio

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _examples_1d_stencil:

===============
Local to remote
===============

When developers write code they typically begin with a simple serial code and
build upon it until all of the required functionality is present. The following
set of examples were developed to demonstrate this iterative process of evolving
a simple serial program to an efficient, fully-distributed |hpx| application. For
this demonstration, we implemented a 1D heat distribution problem. This
calculation simulates the diffusion of heat across a ring from an initialized
state to some user-defined point in the future. It does this by breaking each
portion of the ring into discrete segments and using the current segment's
temperature and the temperature of the surrounding segments to calculate the
temperature of the current segment in the next timestep as shown by
:numref:`1d_stencil_program_flow` below.

.. _1d_stencil_program_flow:

.. figure:: ../_static/images/1d_stencil_program_flow.png

   Heat diffusion example program flow.

We parallelize this code over the following eight examples:

* :download:`Example 1 <../../examples/1d_stencil/1d_stencil_1.cpp>`
* :download:`Example 2 <../../examples/1d_stencil/1d_stencil_2.cpp>`
* :download:`Example 3 <../../examples/1d_stencil/1d_stencil_3.cpp>`
* :download:`Example 4 <../../examples/1d_stencil/1d_stencil_4.cpp>`
* :download:`Example 5 <../../examples/1d_stencil/1d_stencil_5.cpp>`
* :download:`Example 6 <../../examples/1d_stencil/1d_stencil_6.cpp>`
* :download:`Example 7 <../../examples/1d_stencil/1d_stencil_7.cpp>`
* :download:`Example 8 <../../examples/1d_stencil/1d_stencil_8.cpp>`

The first example is straight serial code. In this code we instantiate a vector
``U`` that contains two vectors of doubles as seen in the structure
``stepper``.

.. literalinclude:: ../../examples/1d_stencil/1d_stencil_1.cpp
   :language: c++
   :start-after: //[stepper_1
   :end-before: //]

Each element in the vector of doubles represents a single grid point. To
calculate the change in heat distribution, the temperature of each grid point,
along with its neighbors, is passed to the function ``heat``. In order to
improve readability, references named ``current`` and ``next`` are created
which, depending on the time step, point to the first and second vector of
doubles. The first vector of doubles is initialized with a simple heat ramp.
After calling the heat function with the data in the ``current`` vector, the
results are placed into the ``next`` vector.

In example 2 we employ a technique called futurization. Futurization is a method
by which we can easily transform a code that is serially executed into a code
that creates asynchronous threads. In the simplest case this involves replacing
a variable with a future to a variable, a function with a future to a function,
and adding a ``.get()`` at the point where a value is actually needed. The code
below shows how this technique was applied to the ``struct stepper``.

.. literalinclude:: ../../examples/1d_stencil/1d_stencil_2.cpp
   :language: c++
   :start-after: //[stepper_2
   :end-before: //]

In example 2, we redefine our partition type as a ``shared_future`` and, in
``main``, create the object ``result``, which is a future to a vector of
partitions. We use ``result`` to represent the last vector in a string of
vectors created for each timestep. In order to move to the next timestep, the
values of a partition and its neighbors must be passed to ``heat`` once the
futures that contain them are ready. In |hpx|, we have an LCO (Local Control
Object) named Dataflow that assists the programmer in expressing this
dependency. Dataflow allows us to pass the results of a set of futures to a
specified function when the futures are ready. Dataflow takes three types of
arguments, one which instructs the dataflow on how to perform the function call
(async or sync), the function to call (in this case ``Op``), and futures to the
arguments that will be passed to the function. When called, dataflow immediately
returns a future to the result of the specified function. This allows users to
string dataflows together and construct an execution tree.

After the values of the futures in dataflow are ready, the values must be pulled
out of the future container to be passed to the function ``heat``. In order to
do this, we use the HPX facility ``unwrapping``, which underneath calls
``.get()`` on each of the futures so that the function ``heat`` will be passed
doubles and not futures to doubles.

By setting up the algorithm this way, the program will be able to execute as
quickly as the dependencies of each future are met. Unfortunately, this example
runs terribly slow. This increase in execution time is caused by the overheads
needed to create a future for each data point. Because the work done within each
call to heat is very small, the overhead of creating and scheduling each of the
three futures is greater than that of the actual useful work! In order to
amortize the overheads of our synchronization techniques, we need to be able to
control the amount of work that will be done with each future. We call this
amount of work per overhead grain size.

In example 3, we return to our serial code to figure out how to control the
grain size of our program. The strategy that we employ is to create "partitions"
of data points. The user can define how many partitions are created and how many
data points are contained in each partition. This is accomplished by creating
the ``struct partition``, which contains a member object ``data_``, a vector of
doubles that holds the data points assigned to a particular instance of
``partition``.

In example 4, we take advantage of the partition setup by redefining ``space``
to be a vector of shared_futures with each future representing a partition. In
this manner, each future represents several data points. Because the user can
define how many data points are in each partition, and, therefore, how
many data points are represented by one future, a user can control the
grainsize of the simulation. The rest of the code is then futurized in the same
manner as example 2. It should be noted how strikingly similar
example 4 is to example 2.

Example 4 finally shows good results. This code scales equivalently to the
OpenMP version. While these results are promising, there are more opportunities
to improve the application's scalability. Currently, this code only runs on one
:term:`locality`, but to get the full benefit of |hpx|, we need to be able to
distribute the work to other machines in a cluster. We begin to add this
functionality in example 5.

In order to run on a distributed system, a large amount of boilerplate code must
be added. Fortunately, |hpx| provides us with the concept of a :term:`component`,
which saves us from having to write quite as much code. A component is an object
that can be remotely accessed using its global address. Components are made of
two parts: a server and a client class. While the client class is not required,
abstracting the server behind a client allows us to ensure type safety instead
of having to pass around pointers to global objects. Example 5 renames example
4's ``struct partition`` to ``partition_data`` and adds serialization support.
Next, we add the server side representation of the data in the structure
``partition_server``. ``Partition_server`` inherits from
``hpx::components::component_base``, which contains a server-side component
boilerplate. The boilerplate code allows a component's public members to be
accessible anywhere on the machine via its Global Identifier (GID). To
encapsulate the component, we create a client side helper class. This object
allows us to create new instances of our component and access its members
without having to know its GID. In addition, we are using the client class to
assist us with managing our asynchrony. For example, our client class
``partition``\ 's member function ``get_data()`` returns a future to
``partition_data get_data()``. This struct inherits its boilerplate code from
``hpx::components::client_base``.

In the structure ``stepper``, we have also had to make some changes to
accommodate a distributed environment. In order to get the data from a
particular neighboring partition, which could be remote, we must retrieve the data from all
of the neighboring partitions. These retrievals are asynchronous and the function
``heat_part_data``, which, amongst other things, calls ``heat``, should not be
called unless the data from the neighboring partitions have arrived. Therefore,
it should come as no surprise that we synchronize this operation with another
instance of dataflow (found in ``heat_part``). This dataflow receives futures
to the data in the current and surrounding partitions by calling ``get_data()``
on each respective partition. When these futures are ready, dataflow passes them
to the ``unwrapping`` function, which extracts the shared_array of doubles and
passes them to the lambda. The lambda calls ``heat_part_data`` on the
:term:`locality`, which the middle partition is on.

Although this example could run distributed, it only runs on one
:term:`locality`, as it always uses ``hpx::find_here()`` as the target for the
functions to run on.

In example 6, we begin to distribute the partition data on different nodes. This
is accomplished in ``stepper::do_work()`` by passing the GID of the
:term:`locality` where we wish to create the partition to the partition
constructor.

.. literalinclude:: ../../examples/1d_stencil/1d_stencil_6.cpp
   :language: c++
   :start-after: //[do_work_6
   :end-before: //]

We distribute the partitions evenly based on the number of localities used,
which is described in the function ``locidx``. Because some of the data needed
to update the partition in ``heat_part`` could now be on a new :term:`locality`,
we must devise a way of moving data to the :term:`locality` of the middle
partition. We accomplished this by adding a switch in the function
``get_data()`` that returns the end element of the ``buffer data_`` if it is
from the left partition or the first element of the buffer if the data is from
the right partition. In this way only the necessary elements, not the whole
buffer, are exchanged between nodes. The reader should be reminded that this
exchange of end elements occurs in the function ``get_data()`` and, therefore, is
executed asynchronously.

Now that we have the code running in distributed, it is time to make some
optimizations. The function ``heat_part`` spends most of its time on two tasks:
retrieving remote data and working on the data in the middle partition. Because
we know that the data for the middle partition is local, we can overlap the work
on the middle partition with that of the possibly remote call of ``get_data()``.
This algorithmic change, which was implemented in example 7, can be seen below:

.. literalinclude:: ../../examples/1d_stencil/1d_stencil_7.cpp
   :language: c++
   :start-after: //[stepper_7
   :end-before: //]

Example 8 completes the futurization process and utilizes the full potential of
|hpx| by distributing the program flow to multiple localities, usually defined as
nodes in a cluster. It accomplishes this task by running an instance of |hpx| main
on each :term:`locality`. In order to coordinate the execution of the program,
the ``struct stepper`` is wrapped into a component. In this way, each
:term:`locality` contains an instance of stepper that executes its own instance
of the function ``do_work()``. This scheme does create an interesting
synchronization problem that must be solved. When the program flow was being
coordinated on the head node, the GID of each component was known. However, when
we distribute the program flow, each partition has no notion of the GID of its
neighbor if the next partition is on another :term:`locality`. In order to make
the GIDs of neighboring partitions visible to each other, we created two buffers
to store the GIDs of the remote neighboring partitions on the left and right
respectively. These buffers are filled by sending the GID of newly created
edge partitions to the right and left buffers of the neighboring localities.

In order to finish the simulation, the solution vectors named ``result`` are then
gathered together on :term:`locality` 0 and added into a vector of spaces
``overall_result`` using the |hpx| functions ``gather_id`` and ``gather_here``.

.. todo::

   Insert performance of ``stencil_8``.

Example 8 completes this example series, which takes the serial code of example 1
and incrementally morphs it into a fully distributed parallel code. This
evolution was guided by the simple principles of futurization, the knowledge of
grainsize, and utilization of components. Applying these techniques easily
facilitates the scalable parallelization of most applications.

