..
    Copyright (C) 2007-2014 Hartmut Kaiser
    Copyright (C) 2011 Bryce Lelbach
    Copyright (C) 2013 Pyry Jahkola
    Copyright (C) 2013 Thomas Heller

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _running_on_batch_systems:

========================
Running on batch systems
========================

This section walks you through launching |hpx| applications on various batch
systems.

.. _unix_pbs:

How to use |hpx| applications with PBS
======================================

Most |hpx| applications are executed on parallel computers. These platforms
typically provide integrated job management services that facilitate the
allocation of computing resources for each parallel program. |hpx| includes
support for one of the most common job management systems, the
Portable Batch System (PBS).

All PBS jobs require a script to specify the resource requirements and other
parameters associated with a parallel job. The PBS script is basically a shell
script with PBS directives placed within commented sections at the beginning of
the file. The remaining (not commented-out) portions of the file executes just
like any other regular shell script. While the description of all available PBS
options is outside the scope of this tutorial (the interested reader may refer
to in-depth `documentation <http://www.clusterresources.com/torquedocs21/>`_ for
more information), below is a minimal example to illustrate the approach. The following
test application will use the multithreaded ``hello_world_distributed``
program, explained in the section :ref:`examples_hello_world`.

.. code-block:: bash

   #!/bin/bash
   #
   #PBS -l nodes=2:ppn=4

   APP_PATH=~/packages/hpx/bin/hello_world_distributed
   APP_OPTIONS=

   pbsdsh -u $APP_PATH $APP_OPTIONS --hpx:nodes=`cat $PBS_NODEFILE`

.. caution::

   If the first application specific argument (inside ``$APP_OPTIONS``) is a
   non-option (i.e., does not start with a ``-`` or a ``--``), then the argument has to
   be placed before the option :option:`--hpx:nodes`, which, in this case, should
   be the last option on the command line.

   Alternatively, use the option :option:`--hpx:endnodes` to explicitly mark the
   end of the list of node names:

   .. code-block:: shell-session

      $ pbsdsh -u $APP_PATH --hpx:nodes`cat $PBS_NODEFILE` --hpx:endnodes $APP_OPTIONS

The ``#PBS -l nodes=2:ppn=4`` directive will cause two compute nodes to be
allocated for the application, as specified in the option ``nodes``. Each of the
nodes will dedicate four cores to the program, as per the option ``ppn``, short
for "processors per node" (PBS does not distinguish between processors and
cores). Note that requesting more cores per node than physically available is
pointless and may prevent PBS from accepting the script.

On newer PBS versions the PBS command syntax might be different. For instance,
the PBS script above would look like:

.. code-block:: bash

   #!/bin/bash
   #
   #PBS -l select=2:ncpus=4

   APP_PATH=~/packages/hpx/bin/hello_world_distributed
   APP_OPTIONS=

   pbsdsh -u $APP_PATH $APP_OPTIONS --hpx:nodes=`cat $PBS_NODEFILE`

``APP_PATH`` and ``APP_OPTIONS`` are shell variables that respectively specify
the correct path to the executable (``hello_world_distributed`` in this case)
and the command line options. Since the ``hello_world_distributed`` application
doesn't need any command line options, ``APP_OPTIONS`` has been left empty.
Unlike in other execution environments, there is no need to use the
:option:`--hpx:threads` option to indicate the required number of OS threads per
node; the |hpx| library will derive this parameter automatically from PBS.

Finally, |pbsdsh| is a PBS command that starts tasks to the resources allocated
to the current job. It is recommended to leave this line as shown and modify
only the PBS options and shell variables as needed for a specific application.

.. important::

   A script invoked by |pbsdsh| starts in a very basic environment: the user's
   ``$HOME`` directory is defined and is the current directory, the ``LANG``
   variable is set to ``C`` and the ``PATH`` is set to the basic
   ``/usr/local/bin:/usr/bin:/bin`` as defined in a system-wide file
   pbs_environment. Nothing that would normally be set up by a system shell
   profile or user shell profile is defined, unlike the environment for the main
   job script.

Another choice is for the |pbsdsh| command in your main job script to invoke
your program via a shell, like ``sh`` or ``bash``, so that it gives an initialized
environment for each instance. Users can create a small script ``runme.sh``, which is used
to invoke the program:

.. code-block:: bash

   #!/bin/bash
   # Small script which invokes the program based on what was passed on its
   # command line.
   #
   # This script is executed by the bash shell which will initialize all
   # environment variables as usual.
   $@

Now, the script is invoked using the |pbsdsh| tool:

.. code-block:: bash

   #!/bin/bash
   #
   #PBS -l nodes=2:ppn=4

   APP_PATH=~/packages/hpx/bin/hello_world_distributed
   APP_OPTIONS=

   pbsdsh -u runme.sh $APP_PATH $APP_OPTIONS --hpx:nodes=`cat $PBS_NODEFILE`

All that remains now is submitting the job to the queuing system. Assuming that
the contents of the PBS script were saved in the file ``pbs_hello_world.sh`` in the
current directory, this is accomplished by typing:

.. code-block:: shell-session

   $ qsub ./pbs_hello_world_pbs.sh

If the job is accepted, |qsub| will print out the assigned job ID, which may
look like:

.. code-block:: shell-session

   $ 42.supercomputer.some.university.edu

To check the status of your job, issue the following command:

.. code-block:: shell-session

   $ qstat 42.supercomputer.some.university.edu

and look for a single-letter job status symbol. The common cases include:

* *Q* - signifies that the job is queued and awaiting its turn to be executed.
* *R* - indicates that the job is currently running.
* *C* - means that the job has completed.

The example |qstat| output below shows a job waiting for execution resources
to become available:

.. code-block:: text

   Job id                    Name             User            Time Use S Queue
   ------------------------- ---------------- --------------- -------- - -----
   42.supercomputer          ...ello_world.sh joe_user               0 Q batch

After the job completes, PBS will place two files, ``pbs_hello_world.sh.o42`` and
``pbs_hello_world.sh.e42``, in the directory where the job was submitted. The
first contains the standard output and the second contains the standard error
from all the nodes on which the application executed. In our example, the error
output file should be empty and the standard output file should contain something
similar to:

.. code-block:: text

   hello world from OS-thread 3 on locality 0
   hello world from OS-thread 2 on locality 0
   hello world from OS-thread 1 on locality 1
   hello world from OS-thread 0 on locality 0
   hello world from OS-thread 3 on locality 1
   hello world from OS-thread 2 on locality 1
   hello world from OS-thread 1 on locality 0
   hello world from OS-thread 0 on locality 1

Congratulations! You have just run your first distributed |hpx| application!

.. _unix_slurm:

How to use |hpx| applications with SLURM
========================================

Just like PBS (described in section :ref:`unix_pbs`), |slurm| is a job
management system which is widely used on large supercomputing systems. Any
|hpx| application can easily be run using SLURM. This section describes how this
can be done.

The easiest way to run an |hpx| application using SLURM is to utilize the
command line tool |srun|, which interacts with the SLURM batch scheduling system:

.. code-block:: shell-session

   $ srun -p <partition> -N <number-of-nodes> hpx-application <application-arguments>

Here, ``<partition>`` is one of the node partitions existing on the target
machine (consult the machine's documentation to get a list of existing
partitions) and ``<number-of-nodes>`` is the number of compute nodes that should
be used. By default, the |hpx| application is started with one :term:`locality` per
node and uses all available cores on a node. You can change the number of
localities started per node (for example, to account for NUMA effects) by
specifying the ``-n`` option of srun. The number of cores per :term:`locality`
can be set by ``-c``. The ``<application-arguments>`` are any application
specific arguments that need to be passed on to the application.

.. note::

   There is no need to use any of the |hpx| command line options related to the
   number of localities, number of threads, or related to networking ports. All
   of this information is automatically extracted from the SLURM environment by
   the |hpx| startup code.

.. important::

   The |srun| documentation explicitly states: "If ``-c`` is specified without
   ``-n``, as many tasks will be allocated per node as possible while satisfying
   the ``-c`` restriction. For instance on a cluster with 8 CPUs per node, a job
   request for 4 nodes and 3 CPUs per task may be allocated 3 or 6 CPUs per node
   (1 or 2 tasks per node) depending upon resource consumption by other jobs."
   For this reason, it's recommended to always specify ``-n <number-of-instances>``,
   even if ``<number-of-instances>`` is equal to one (``1``).

Interactive shells
------------------

To get an interactive development shell on one of the nodes, users can issue the
following command:

.. code-block:: shell-session

   $ srun -p <node-type> -N <number-of-nodes> --pty /bin/bash -l

After the shell has been opened, users can run their |hpx| application. By default,
it uses all available cores. Note that if you requested one node, you don't need
to do ``srun`` again. However, if you requested more than one node, and want to
run your distributed application, you can use ``srun`` again to start up the
distributed |hpx| application. It will use the resources that have been requested
for the interactive shell.

Scheduling batch jobs
---------------------

The above mentioned method of running |hpx| applications is fine for development
purposes. The disadvantage that comes with ``srun`` is that it only returns once
the application is finished. This might not be appropriate for longer-running
applications (for example, benchmarks or larger scale simulations). In order to
cope with that limitation, users can use the |sbatch| command.

The ``sbatch`` command expects a script that it can run once the requested
resources are available. In order to request resources, users need to add
``#SBATCH`` comments in their script or provide the necessary parameters to
``sbatch`` directly. The parameters are the same as with ``run``. The commands
you need to execute are the same you would need to start your application as if
you were in an interactive shell.

