//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/mpi.hpp>
#include <hpx/program_options.hpp>
#include <hpx/runtime/threads/policies/scheduler_base.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
#include <hpx/runtime/threads/thread_data.hpp>

#include <array>
#include <atomic>
#include <iostream>
#include <sstream>

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

    if (size < 2)
    {
        std::cout << "This test requires N > 1 ranks" << std::endl;
        return hpx::finalize();
    }

    if (rank == 0)
    {
        std::cout << "Rank " << std::setfill(' ') << std::setw(3) << rank
                  << " of " << std::setfill(' ') << std::setw(3) << size
                  << std::endl;
    }

    {
        // this needs to scope all uses of hpx::mpi::executor
        hpx::mpi::enable_user_polling enable_polling;

        // Ring send/recv around N ranks
        // Rank 0      : Send then Recv
        // Rank 1->N-1 : Recv then Send

        hpx::mpi::executor exec(MPI_COMM_WORLD);

        unsigned int const n_loops = 20;
        std::atomic<int> counter(n_loops);
        std::array<int, n_loops> tokens;
        for (unsigned int i = 0; i != n_loops; ++i)
        {
            tokens[i] = (rank == 0) ? 1 : -1;
            int rank_from = (size + rank - 1) % size;
            int rank_to = (rank + 1) % size;

            // all ranks pre-post a receive
            hpx::future<int> f_recv =
                hpx::async(exec, MPI_Irecv, &tokens[i], 1, MPI_INT, rank_from, i);

            // when the recv completes,
            f_recv.then([=, &tokens, &counter](auto&&) {
                msg_recv(rank, size, rank_to, rank_from, tokens[i], i);
                if (rank > 0)
                {
                    // send the incremented token to the next rank
                    ++tokens[i];
                    hpx::future<void> f_send = hpx::async(
                        exec, MPI_Isend, &tokens[i], 1, MPI_INT, rank_to, i);
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
                auto f_send =
                    hpx::async(exec, MPI_Isend, &tokens[i], 1, MPI_INT, rank_to, i);
                f_send.then([=, &tokens, &counter](auto&&) {
                    msg_send(rank, size, rank_to, rank_from, tokens[i], i);
                });
            }
        }

        // Our simple counter should reach zero when all send/recv pairs are done
        enable_polling.wait([&]() { return counter != 0; });

        // let the user polling go out of scope
    }

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
    // Init MPI
    int provided = MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE)
    {
        std::cout << "Provided MPI is not : MPI_THREAD_MULTIPLE " << provided
                  << std::endl;
    }

    int result = hpx::init(argc, argv);

    // Finalize MPI
    MPI_Finalize();

    return result;
}
