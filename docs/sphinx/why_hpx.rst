..
    Copyright (C) 2007-2013 Hartmut Kaiser

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _why_hpx:

==========
Why |hpx|?
==========

Current advances in high performance computing (HPC) continue to suffer from the
issues plaguing parallel computation. These issues include, but are not limited
to, ease of programming, inability to handle dynamically changing workloads,
scalability, and efficient utilization of system resources. Emerging
technological trends such as multi-core processors further highlight limitations
of existing parallel computation models. To mitigate the aforementioned
problems, it is necessary to rethink the approach to parallelization models.
ParalleX contains mechanisms such as multi-threading, :term:`parcels <parcel>`,
:term:`global name space <AGAS>` support, percolation and :term:`local control
object`\ s (:term:`LCO`). By design, ParalleX overcomes limitations of current
models of parallelism by alleviating contention, latency, overhead and
starvation. With ParalleX, it is further possible to increase performance by at
least an order of magnitude on challenging parallel algorithms, e.g., dynamic
directed graph algorithms and adaptive mesh refinement methods for astrophysics.
An additional benefit of ParalleX is fine-grained control of power usage,
enabling reductions in power consumption.

ParalleX---a new execution model for future architectures
---------------------------------------------------------

ParalleX is a new parallel execution model that offers an alternative to
the conventional computation models, such as message passing. ParalleX
distinguishes itself by:

* Split-phase transaction model
* Message-driven
* Distributed shared memory (not cache coherent)
* Multi-threaded
* Futures synchronization
* :term:`Local Control Object`\ s (:term:`LCO`\ s)
* Synchronization for anonymous producer-consumer scenarios
* Percolation (pre-staging of task data)

The ParalleX model is intrinsically latency hiding, delivering an abundance of
variable-grained parallelism within a hierarchical namespace environment. The
goal of this innovative strategy is to enable future systems delivering very
high efficiency, increased scalability and ease of programming. ParalleX can
contribute to significant improvements in the design of all levels of computing
systems and their usage from application algorithms and their programming
languages to system architecture and hardware design together with their
supporting compilers and operating system software.

What is |hpx|?
--------------

High Performance ParalleX (|hpx|) is the first runtime system implementation of
the ParalleX execution model. The |hpx| runtime software package is a modular,
feature-complete, and performance-oriented representation of the ParalleX
execution model targeted at conventional parallel computing architectures, such
as SMP nodes and commodity clusters. It is academically developed and freely
available under an open source license. We provide |hpx| to the community for
experimentation and application to achieve high efficiency and scalability for
dynamic adaptive and irregular computational problems. |hpx| is a C++ library
that supports a set of critical mechanisms for dynamic adaptive resource
management and lightweight task scheduling within the context of a global
address space. It is solidly based on many years of experience in writing highly
parallel applications for HPC systems.

The two-decade success of the communicating sequential processes (CSP) execution
model and its message passing interface (MPI) programming model have been
seriously eroded by challenges of power, processor core complexity, multi-core
sockets, and heterogeneous structures of GPUs. Both efficiency and scalability
for some current (strong scaled) applications and future Exascale applications
demand new techniques to expose new sources of algorithm parallelism and exploit
unused resources through adaptive use of runtime information.

The ParalleX execution model replaces CSP to provide a new computing paradigm
embodying the governing principles for organizing and conducting highly
efficient scalable computations greatly exceeding the capabilities of today's
problems. |hpx| is the first practical, reliable, and performance-oriented
runtime system incorporating the principal concepts of the ParalleX model
publicly provided in open source release form.

|hpx| is designed by the |stellar|_ Group (**S**\ ystems **T**\ echnology,
**E**\ mergent Para\ **ll**\ elism, and **A**\ lgorithm **R**\ esearch) at
|lsu|_'s |cct|_ to enable developers to exploit the full processing power of
many-core systems with an unprecedented degree of parallelism. |stellar|_ is a
research group focusing on system software solutions and scientific application
development for hybrid and many-core hardware architectures.

For more information about the |stellar|_ Group, see :ref:`people`.

What makes our systems slow?
----------------------------

Estimates say that we currently run our computers at well below 100% efficiency.
The theoretical peak performance (usually measured in
`FLOPS <http://en.wikipedia.org/wiki/FLOPS>`_---floating point operations per
second) is much higher than any practical peak performance reached by any
application. This is particularly true for highly parallel hardware. The more
hardware parallelism we provide to an application, the better the application
must scale in order to efficiently use all the resources of the machine. Roughly
speaking, we distinguish two forms of scalability: strong scaling (see
`Amdahl's Law <http://en.wikipedia.org/wiki/Amdahl%27s_law>`_) and weak scaling
(see `Gustafson's Law <http://en.wikipedia.org/wiki/Gustafson%27s_law>`_). Strong
scaling is defined as how the solution time varies with the number of processors
for a fixed **total** problem size. It gives an estimate of how much faster we
can solve a particular problem by throwing more resources at it. Weak scaling is
defined as how the solution time varies with the number of processors for a
fixed problem size **per processor**. In other words, it defines how much more
data can we process by using more hardware resources.

In order to utilize as much hardware parallelism as possible an application must
exhibit excellent strong and weak scaling characteristics, which requires a high
percentage of work executed in parallel, i.e., using multiple threads of
execution. Optimally, if you execute an application on a hardware resource with
N processors it either runs N times faster or it can handle N times more data.
Both cases imply 100% of the work is executed on all available processors in
parallel. However, this is just a theoretical limit. Unfortunately, there are
more things that limit scalability, mostly inherent to the hardware
architectures and the programming models we use. We break these limitations into
four fundamental factors that make our systems *SLOW*:

* **S**\ tarvation occurs when there is insufficient concurrent work available to
  maintain high utilization of all resources.
* **L**\ atencies are imposed by the time-distance delay intrinsic to accessing
  remote resources and services.
* **O**\ verhead is work required for the management of parallel actions and
  resources on the critical execution path, which is not necessary in a
  sequential variant.
* **W**\ aiting for contention resolution is the delay due to the lack of
  availability of oversubscribed shared resources.

Each of those four factors manifests itself in multiple and different ways; each
of the hardware architectures and programming models expose specific forms.
However, the interesting part is that all of them are limiting the scalability of
applications no matter what part of the hardware jungle we look at. Hand-helds,
PCs, supercomputers, or the cloud, all suffer from the reign of the 4 horsemen:
**S**\ tarvation, **L**\ atency, **O**\ verhead, and **C**\ ontention. This
realization is very important as it allows us to derive the criteria for
solutions to the scalability problem from first principles, and it allows us to
focus our analysis on very concrete patterns and measurable metrics. Moreover,
any derived results will be applicable to a wide variety of targets.

Technology demands new response
-------------------------------

Today's computer systems are designed based on the initial ideas of
`John von Neumann <http://qss.stanford.edu/~godfrey/vonNeumann/vnedvac.pdf>`_, as
published back in 1945, and later extended by the
`Harvard architecture <http://en.wikipedia.org/wiki/Harvard_architecture>`_. These
ideas form the foundation, the execution model, of computer systems we use
currently. However, a new response is required in the light of the demands
created by today's technology.

So, what are the overarching objectives for designing systems allowing for
applications to scale as they should? In our opinion, the main objectives are:

* Performance: as previously mentioned, scalability and efficiency are the main criteria
  people are interested in.
* Fault tolerance: the low expected mean time between failures (`MTBF
  <http://en.wikipedia.org/wiki/Mean_time_between_failures>`_) of future systems
  requires embracing faults, not trying to avoid them.
* Power: minimizing energy consumption is a must as it is one of the major cost
  factors today, and will continue to rise in the future.
* Generality: any system should be usable for a broad set of use cases.
* Programmability: for programmer this is a very important objective,
  ensuring long term platform stability and portability.

What needs to be done to meet those objectives, to make applications scale
better on tomorrow's architectures? Well, the answer is almost obvious: we need
to devise a new execution model---a set of governing principles for the holistic
design of future systems---targeted at minimizing the effect of the outlined
**SLOW** factors. Everything we create for future systems, every design decision
we make, every criteria we apply, have to be validated against this single,
uniform metric. This includes changes in the hardware architecture we
prevalently use today, and it certainly involves new ways of writing software,
starting from the operating system, runtime system, compilers, and at the
application level. However, the key point is that all those layers have to be
co-designed; they are interdependent and cannot be seen as separate facets. The
systems we have today have been evolving for over 50 years now. All layers
function in a certain way, relying on the other layers to do so. But we do not
have the time to wait another 50 years for a new coherent system to evolve.
The new paradigms are needed now---therefore, co-design is the key.

Governing principles applied while developing |hpx|
---------------------------------------------------

As it turn out, we do not have to start from scratch. Not everything has to be
invented and designed anew. Many of the ideas needed to combat the 4 horsemen
already exist, many for more than 30 years. All it takes is to gather
them into a coherent approach. We'll highlight some of the derived principles we
think to be crucial for defeating **SLOW**. Some of those are focused on
high-performance computing, others are more general.

Focus on latency hiding instead of latency avoidance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is impossible to design a system exposing zero latencies. In an effort to
come as close as possible to this goal many optimizations are mainly targeted
towards minimizing latencies. Examples for this can be seen everywhere, such as
low latency network technologies like `InfiniBand
<http://en.wikipedia.org/wiki/InfiniBand>`_, caching memory hierarchies in all
modern processors, the constant optimization of existing |mpi|_ implementations
to reduce related latencies, or the data transfer latencies intrinsic to the way
we use `GPGPUs <http://en.wikipedia.org/wiki/GPGPU>`_ today. It is important to
note that existing latencies are often tightly related to some resource having
to wait for the operation to be completed. At the same time it would be
perfectly fine to do some other, unrelated work in the meantime, allowing the
system to hide the latencies by filling the idle-time with useful work. Modern
systems already employ similar techniques (pipelined instruction execution in
the processor cores, asynchronous input/output operations, and many more). What
we propose is to go beyond anything we know today and to make latency hiding an
intrinsic concept of the operation of the whole system stack.

Embrace fine-grained parallelism instead of heavyweight threads
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If we plan to hide latencies even for very short operations, such as fetching
the contents of a memory cell from main memory (if it is not already cached), we
need to have very lightweight threads with extremely short context switching
times, optimally executable within one cycle. Granted, for mainstream
architectures, this is not possible today (even if we already have special
machines supporting this mode of operation, such as the `Cray XMT
<http://en.wikipedia.org/wiki/Cray_XMT>`_). For conventional systems, however,
the smaller the overhead of a context switch and the finer the granularity of
the threading system, the better will be the overall system utilization and its
efficiency. For today's architectures we already see a flurry of libraries
providing exactly this type of functionality: non-pre-emptive, task-queue based
parallelization solutions, such as |tbb|_, |ppl|_, |cilk_pp|_, and many others.
The possibility to suspend a current task if some preconditions for its
execution are not met (such as waiting for I/O or the result of a different
task), seamlessly switching to any other task which can continue, and to
reschedule the initial task after the required result has been calculated, which
makes the implementation of latency hiding almost trivial.

Rediscover constraint-based synchronization to replace global barriers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code we write today is riddled with implicit (and explicit) global barriers.
By "global barriers," we mean the synchronization of the control flow between
several (very often all) threads (when using |openmp|_) or processes (|mpi|_).
For instance, an implicit global barrier is inserted after each loop
parallelized using |openmp|_ as the system synchronizes the threads used to
execute the different iterations in parallel. In |mpi|_ each of the
communication steps imposes an explicit barrier onto the execution flow as
(often all) nodes have to be synchronized. Each of those barriers is like the eye
of a needle the overall execution is forced to be squeezed through. Even
minimal fluctuations in the execution times of the parallel threads (jobs)
causes them to wait. Additionally, it is often only one of the executing threads 
that performs the actual reduce operation, which further impedes parallelism. A closer
analysis of a couple of key algorithms used in science applications reveals that
these global barriers are not always necessary. In many cases it is sufficient
to synchronize a small subset of the threads. Any operation should proceed
whenever the preconditions for its execution are met, and only those. Usually
there is no need to wait for iterations of a loop to finish before you can
continue calculating other things; all you need is to complete the iterations
that produce the required results for the next operation. Good
bye global barriers, hello constraint based synchronization! People have been
trying to build this type of computing (and even computers) since the 1970s.
The theory behind what they did is based on ideas around static and
dynamic dataflow. There are certain attempts today to get back to those ideas
and to incorporate them with modern architectures. For instance, a lot of work
is being done in the area of constructing dataflow-oriented execution trees. Our
results show that employing dataflow techniques in combination with the other
ideas, as outlined herein, considerably improves scalability for many problems.

Adaptive locality control instead of static data distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While this principle seems to be a given for single desktop or laptop computers
(the operating system is your friend), it is everything but ubiquitous on modern
supercomputers, which are usually built from a large number of separate nodes
(i.e., Beowulf clusters), tightly interconnected by a high-bandwidth, low-latency
network. Today's prevalent programming model for those is MPI, which does not
directly help with proper data distribution, leaving it to the programmer to
decompose the data to all of the nodes the application is running on. There are
a couple of specialized languages and programming environments based on |pgas|_
(Partitioned Global Address Space) designed to overcome this limitation, such as
|chapel|_, |x10|_, |upc|_, or |fortress|_. However, all systems based on PGAS
rely on static data distribution. This works fine as long as this static data
distribution does not result in heterogeneous workload distributions or other
resource utilization imbalances. In a distributed system these imbalances can be
mitigated by migrating part of the application data to different localities
(nodes). The only framework supporting (limited) migration today is |charm_pp|_.
The first attempts towards solving related problem go back decades as well, a
good example is the `Linda coordination language
<http://en.wikipedia.org/wiki/Linda_(coordination_language)>`_. Nevertheless,
none of the other mentioned systems support data migration today, which forces
the users to either rely on static data distribution and live with the related
performance hits or to implement everything themselves, which is very tedious
and difficult. We believe that the only viable way to flexibly support dynamic
and adaptive :term:`locality` control is to provide a global, uniform address
space to the applications, even on distributed systems.

Prefer moving work to the data over moving data to the work
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the best performance it seems obvious to minimize the amount of bytes
transferred from one part of the system to another. This is true on all levels.
At the lowest level we try to take advantage of processor memory caches, thus,
minimizing memory latencies. Similarly, we try to amortize the data transfer
time to and from `GPGPUs <http://en.wikipedia.org/wiki/GPGPU>`_ as much as
possible. At high levels we try to minimize data transfer between different
nodes of a cluster or between different virtual machines on the cloud. Our
experience (well, it's almost common wisdom) shows that the amount of bytes
necessary to encode a certain operation is very often much smaller than the
amount of bytes encoding the data the operation is performed upon. Nevertheless,
we still often transfer the data to a particular place where we execute the
operation just to bring the data back to where it came from afterwards. As an
example let's look at the way we usually write our applications for clusters
using MPI. This programming model is all about data transfer between nodes.
MPI is the prevalent programming model for clusters, and it is fairly
straightforward to understand and to use. Therefore, we often write 
applications in a way that accommodates this model, centered around data transfer.
These applications usually work well for smaller problem sizes and for regular
data structures. The larger the amount of data we have to churn and the more
irregular the problem domain becomes, the worse the overall machine
utilization and the (strong) scaling characteristics become. While it is not impossible
to implement more dynamic, data driven, and asynchronous applications using
MPI, it is somewhat difficult to do so. At the same time, if we look at
applications that prefer to execute the code close to the :term:`locality` where the
data was placed, i.e., utilizing active messages (for instance based on
|charm_pp|_), we see better asynchrony, simpler application codes, and improved
scaling.

Favor message driven computation over message passing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Today's prevalently used programming model on parallel (multi-node) systems is
MPI. It is based on message passing, as the name implies, which means that
the receiver has to be aware of a message about to come in. Both codes, the
sender and the receiver, have to synchronize in order to perform the
communication step. Even the newer, asynchronous interfaces require explicitly
coding the algorithms around the required communication scheme. As a result, everything
but the most trivial MPI applications spends a considerable amount of time
waiting for incoming messages, thus, causing starvation and latencies to impede
full resource utilization. The more complex and more dynamic the data structures
and algorithms become, the larger the adverse effects. The community discovered
message-driven and data-driven methods of implementing algorithms a long
time ago, and systems such as |charm_pp|_ have already integrated active
messages demonstrating the validity of the concept. Message-driven computation
allows for sending messages without requiring the receiver to actively wait for
them. Any incoming message is handled asynchronously and triggers the encoded
action by passing along arguments and---possibly---continuations. |hpx| combines
this scheme with work-queue based scheduling as described above, which allows
the system to almost completely overlap any communication with useful work,
thereby minimizing latencies.
