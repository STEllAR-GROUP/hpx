//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/program_options.hpp>
//
#include <hpx/mpi/mpi_future.hpp>
//
#include <iostream>
#include <sstream>
#include <array>
#include <atomic>
//
#include <mpi.h>

void msg_recv(int rank, int size, int /*to*/, int from, int token, unsigned tag)
{
    std::ostringstream temp;
    temp << "Rank " << std::setfill(' ') << std::setw(3) << rank << " of "
         << std::setfill(' ') << std::setw(3) << size << " Recv token "
         << std::setfill(' ') << std::setw(3) << token << " from rank "
         << std::setfill(' ') << std::setw(3) << from << " tag "
         << std::setfill(' ') << std::setw(3) << tag;
    std::cout << temp.str() << std::endl;
}

void msg_send(int rank, int size, int to, int /*from*/, int token, unsigned tag)
{
    std::ostringstream temp;
    temp << "Rank " << std::setfill(' ') << std::setw(3) << rank << " of "
         << std::setfill(' ') << std::setw(3) << size << " Sent token "
         << std::setfill(' ') << std::setw(3) << token << " to   rank "
         << std::setfill(' ') << std::setw(3) << to << " tag "
         << std::setfill(' ') << std::setw(3) << tag;
    std::cout << temp.str() << std::endl;
}

// this is called on an hpx thread after the runtime starts up
int hpx_main(hpx::program_options::variables_map& vm)
{
    int rank, size;
    //
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (size<2) {
        std::cout << "This test requires N>1 ranks" << std::endl;
        return hpx::finalize();
    }
    //
    // tell the scheduler that we want to handle mpi in the background
    // here we use the provided hpx::mpi::poll function but a user
    // provided function or lambda may be supplied
    //
    auto const sched = hpx::threads::get_self_id_data()->get_scheduler_base();
    sched->set_user_polling_function(&hpx::mpi::poll);
    sched->add_scheduler_mode(hpx::threads::policies::enable_user_polling);
    if (rank == 0)
    {
        std::cout << "Rank " << std::setfill(' ') << std::setw(3) << rank
                  << " of " << std::setfill(' ') << std::setw(3) << size
                  << " scheduler is " << sched->get_description() << "\n\n"
                  << std::endl;
    }

    // Ring send/recv around N ranks
    // Rank 0      : Send then Recv
    // Rank 1->N-1 : Recv then Send

    const int n_loops = 20;
    std::atomic<int> counter = n_loops;
    std::array<int, n_loops> tokens;
    for (unsigned int i = 0; i < n_loops; ++i)
    {
        tokens[i] = (rank == 0) ? 1 : -1;
        int rank_from = (size + rank - 1) % size;
        int rank_to = (rank + 1) % size;

        // all ranks pre-post a receive
        hpx::future<int> f_recv = hpx::mpi::async(
            MPI_Irecv, &tokens[i], 1, MPI_INT, rank_from, i, MPI_COMM_WORLD);

        // when the recv completes,
        f_recv.then([=, &tokens, &counter](auto&&) {
            msg_recv(rank, size, rank_to, rank_from, tokens[i], i);
            if (rank > 0)
            {
                // send the incremented token to the next rank
                ++tokens[i];
                hpx::future<void> f_send = hpx::mpi::async(MPI_Isend,
                    &tokens[i], 1, MPI_INT, rank_to, i, MPI_COMM_WORLD);
                // when the send completes
                f_send.then([=, &tokens, &counter](auto&&) {
                    msg_send(rank, size, rank_to, rank_from, tokens[i], i);
                    // ranks>0 are done when they have sent their token
                    --counter;
                });
            }
            else
            {
                // rank 0 is done when it receives its token
                --counter;
            }
        });
        // rank 0 starts the process with a send
        if (rank == 0)
        {
            auto f_send = hpx::mpi::async(
                MPI_Isend, &tokens[i], 1, MPI_INT, rank_to, i, MPI_COMM_WORLD);
            f_send.then([=, &tokens, &counter](auto&&) {
                msg_send(rank, size, rank_to, rank_from, tokens[i], i);
            });
        }
    }

    // Our simple counter should reach zero when all send/recv pairs are done
    while (counter != 0)
    {
        hpx::this_thread::yield();
    }

    //
    // Before exiting, shut down the mpi/user polling loop
    //
    sched->remove_scheduler_mode(hpx::threads::policies::enable_user_polling);

    // This is needed to make sure that one rank does not shut down
    // before others have completed. MPI does not handle that well.
    MPI_Barrier(MPI_COMM_WORLD);
    return hpx::finalize();
}

// the normal int main function that is called at startup and runs on an OS thread
// the user must call hpx::init to start the hpx runtime which will execute hpx_main
// on an hpx thread
int main(int argc, char* argv[])
{
    //
    // Init MPI
    //
    int provided = MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE)
    {
        std::cout << "Provided MPI is not : MPI_THREAD_MULTIPLE " << provided
                  << std::endl;
    }
    return hpx::init(argc, argv);
    //
    // Finalize MPI
    //
    MPI_Finalize();
}
