//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/futures/future.hpp>
#include <hpx/modules/async_local.hpp>
#include <hpx/modules/async_mpi.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/modules/testing.hpp>

#include <array>
#include <atomic>
#include <iostream>
#include <sstream>

#include <mpi.h>

// This test send a message from rank 0 to rank 1 and from R1->R2 in a ring
// until the last rank which sends it back to R0.and completes an iteration
//
// For benchmarking the test has been extended to allow many iterations
// but unfortunately, if we prepost too many receives, MPI has problems
// so we do 1000 iterations per main loop and another loop around that.

using hpx::program_options::options_description;
using hpx::program_options::value;
using hpx::program_options::variables_map;

static bool output = true;

void msg_recv(int rank, int size, int /*to*/, int from, int token, unsigned tag)
{
    // to reduce string corruption on stdout from multiple threads
    // writing simultaneously, we use a stringstream as a buffer
    if (output)
    {
        std::ostringstream temp;
        temp << "Rank " << std::setfill(' ') << std::setw(3) << rank << " of "
             << std::setfill(' ') << std::setw(3) << size << " Recv token "
             << std::setfill(' ') << std::setw(3) << token << " from rank "
             << std::setfill(' ') << std::setw(3) << from << " tag "
             << std::setfill(' ') << std::setw(3) << tag;
        std::cout << temp.str() << std::endl;
    }
}

void msg_send(int rank, int size, int to, int /*from*/, int token, unsigned tag)
{
    if (output)
    {
        std::ostringstream temp;
        temp << "Rank " << std::setfill(' ') << std::setw(3) << rank << " of "
             << std::setfill(' ') << std::setw(3) << size << " Sent token "
             << std::setfill(' ') << std::setw(3) << token << " to   rank "
             << std::setfill(' ') << std::setw(3) << to << " tag "
             << std::setfill(' ') << std::setw(3) << tag;
        std::cout << temp.str() << std::endl;
    }
}

// this is called on an hpx thread after the runtime starts up
int hpx_main(hpx::program_options::variables_map& vm)
{
    int rank, size;
    //
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // if comm size < 2 this test should fail
    // it needs to run on N>2 ranks to be useful
    HPX_TEST_MSG(size > 1, "This test requires N>1 mpi ranks");

    const std::uint64_t iterations = vm["iterations"].as<std::uint64_t>();
    //
    output = vm.count("output") != 0;

    if (rank == 0 && output)
    {
        std::cout << "Rank " << std::setfill(' ') << std::setw(3) << rank
                  << " of " << std::setfill(' ') << std::setw(3) << size
                  << std::endl;
    }

    {
        // this needs to scope all uses of hpx::mpi::experimental::executor
        hpx::mpi::experimental::enable_user_polling enable_polling;

        // Ring send/recv around N ranks
        // Rank 0      : Send then Recv
        // Rank 1->N-1 : Recv then Send

        hpx::mpi::experimental::executor exec(MPI_COMM_WORLD);

        // mpi chokes if we put too many messages into the system at once
        // so do iterations in batches of 1000
        // using an inner and outer loop
        const std::uint64_t max_tokens = 1000;
        std::uint64_t subloops = 1 + iterations / max_tokens;
        std::uint64_t remainder = iterations;

        std::vector<int> tokens(max_tokens, -1);

        hpx::util::high_resolution_timer t;

        for (std::uint64_t j = 0; (j != subloops) && (remainder > 0); ++j)
        {
            std::atomic<int> counter(
                (std::min)(std::uint64_t(max_tokens), remainder));
            for (std::uint64_t i = 0; (i != max_tokens) && (remainder > 0); ++i)
            {
                tokens[i] = (rank == 0) ? 1 : -1;
                int rank_from = (size + rank - 1) % size;
                int rank_to = (rank + 1) % size;

                // all ranks pre-post a receive
                hpx::future<int> f_recv = hpx::async(
                    exec, MPI_Irecv, &tokens[i], 1, MPI_INT, rank_from, i);

                // when the recv completes,
                f_recv.then([=, &tokens, &counter](auto&&) {
                    msg_recv(rank, size, rank_to, rank_from, tokens[i], i);
                    if (rank > 0)
                    {
                        // send the incremented token to the next rank
                        ++tokens[i];
                        hpx::future<int> f_send = hpx::async(exec, MPI_Isend,
                            &tokens[i], 1, MPI_INT, rank_to, i);
                        // when the send completes
                        f_send.then([=, &tokens, &counter](auto&&) {
                            msg_send(
                                rank, size, rank_to, rank_from, tokens[i], i);
                            // ranks > 0 are done when they have sent their token
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
                    auto f_send = hpx::async(
                        exec, MPI_Isend, &tokens[i], 1, MPI_INT, rank_to, i);
                    f_send.then([=, &tokens](auto&&) {
                        msg_send(rank, size, rank_to, rank_from, tokens[i], i);
                    });
                }
                --remainder;
            }

            // Our simple counter should reach zero when all send/recv pairs are done
            hpx::mpi::experimental::wait([&]() { return counter != 0; });
            if (rank == 0 && output)
            {
                std::cout << "remainder " << remainder << " j " << j
                          << std::endl;
            }
        }

        if (rank == 0)
        {
            std::cout << "time " << t.elapsed() << std::endl;
        }

        // let the user polling go out of scope
    }
    return hpx::finalize();
}

// the normal int main function that is called at startup and runs on an OS thread
// the user must call hpx::init to start the hpx runtime which will execute hpx_main
// on an hpx thread
int main(int argc, char* argv[])
{
    // if this test is run with distributed runtime, we need to make sure
    // that all ranks run their main function
    std::vector<std::string> cfg = {"hpx.run_hpx_main!=1"};

    // Init MPI
    int provided = MPI_THREAD_MULTIPLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE)
    {
        std::cout << "Provided MPI is not : MPI_THREAD_MULTIPLE " << provided
                  << std::endl;
    }

    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()(
        "iterations",
        value<std::uint64_t>()->default_value(5000),
        "number of iterations to test")

        ("output", "display messages during test");
    // clang-format on

    // Initialize and run HPX.
    auto result = hpx::init(cmdline, argc, argv, cfg);

    // Finalize MPI
    MPI_Finalize();

    return result || hpx::util::report_errors();
}
