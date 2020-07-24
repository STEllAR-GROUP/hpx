..
    Copyright (c) 2019 The STE||AR-Group

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _libs_mpi:

===
async_mpi
===

The MPI library is intended to simplify the process of integrating MPI based
codes with the |hpx| runtime. Any MPI function that is asynchronous and uses an
MPI_Request may be converted into an hpx::future.
The syntax is designed to allow a simple replacement of the MPI call with a futurized
async version that accepts an executor instead of a communicator,
and returns a future instead of assigning a request.
Typically, an MPI call of the form

.. code-block:: c++

    int MPI_Isend(buf, count, datatype, rank, tag, comm, request);

becomes

.. code-block:: c++

    hpx::future<int> f = hpx::async(executor, MPI_Isend, buf, count, datatype, rank, tag);

When the MPI operation is complete, the future will become ready.
This allows communication to integrated cleanly with the rest of HPX, in particular
the continuation style of programming may be used to build up more
complex code. Consider the following example, that chains user processing,
sends and receives using continuations...

.. code-block:: c++

    // create an executor for MPI dispatch
    hpx::mpi::experimental::executor exec(MPI_COMM_WORLD);

    // post an asynchronous receive using MPI_Irecv
    hpx::future<int> f_recv = hpx::async(
        exec, MPI_Irecv, &data, rank, MPI_INT, rank_from, i);

    // attach a continuation to run when the recv completes,
    f_recv.then([=, &tokens, &counter](auto&&)
    {
        // call an application specific function
        msg_recv(rank, size, rank_to, rank_from, tokens[i], i);

        // send a new message
        hpx::future<int> f_send = hpx::async(
            exec, MPI_Isend, &tokens[i], 1, MPI_INT, rank_to, i);

        // when that send completes
        f_send.then([=, &tokens, &counter](auto&&)
        {
            // call an application specific function
            msg_send(rank, size, rank_to, rank_from, tokens[i], i);
        });
    }

The example above makes use of ``MPI_Isend`` and ``MPI_Irecv``, but *any* MPI function
that uses requests may be futurized in this manner.
The following is a (non exhaustive) list of MPI functions that *should* be supported,
though not all have been tested at the time of writing
(please report any problems to the issue tracker).

.. code-block:: c++

    int MPI_Isend(...);
    int MPI_Ibsend(...);
    int MPI_Issend(...);
    int MPI_Irsend(...);
    int MPI_Irecv(...);
    int MPI_Imrecv(...);
    int MPI_Ibarrier(...);
    int MPI_Ibcast(...);
    int MPI_Igather(...);
    int MPI_Igatherv(...);
    int MPI_Iscatter(...);
    int MPI_Iscatterv(...);
    int MPI_Iallgather(...);
    int MPI_Iallgatherv(...);
    int MPI_Ialltoall(...);
    int MPI_Ialltoallv(...);
    int MPI_Ialltoallw(...);
    int MPI_Ireduce(...);
    int MPI_Iallreduce(...);
    int MPI_Ireduce_scatter(...);
    int MPI_Ireduce_scatter_block(...);
    int MPI_Iscan(...);
    int MPI_Iexscan(...);
    int MPI_Ineighbor_allgather(...);
    int MPI_Ineighbor_allgatherv(...);
    int MPI_Ineighbor_alltoall(...);
    int MPI_Ineighbor_alltoallv(...);
    int MPI_Ineighbor_alltoallw(...);

Note that the |hpx| mpi futurization wrapper should work with *any* asynchronous
`MPI` call, as long as the function signature has the last two arguments
`MPI_xxx(..., MPI_Comm comm, MPI_Request *request)`
- internally these two parameters will be substituted by the executor and future data
parameters that are supplied by template instantiations inside the `hpx::mpi` code.

See the :ref:`API reference <libs_mpi_api>` of this module for more
details.
