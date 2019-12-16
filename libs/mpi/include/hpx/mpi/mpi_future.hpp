//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LIBS_MPI_FUTURE_HPP
#define HPX_LIBS_MPI_FUTURE_HPP

#include <mpi.h>
//
#include <cstdio>
#include <iostream>
#include <array>
#include <utility>
#include <tuple>
#include <list>
#include <mutex>
//
#include <hpx/local_lcos/promise.hpp>
#include <hpx/util/yield_while.hpp>
#include <hpx/memory/intrusive_ptr.hpp>
#include <hpx/debugging/print.hpp>
//

namespace hpx { namespace mpi {

    using  print_on = hpx::debug::enable_print<false>;
    static print_on mpi_debug("MPI_FUTURE");

    // mutex needed to protect mpi request list
    static std::mutex list_mtx_;

    // MPI request object list backed by mpi::future_data
    struct future_data;
    using  future_data_ptr = memory::intrusive_ptr<future_data>;
    static std::list<future_data_ptr> active_requests_;

    // -----------------------------------------------------------------
    // An implementation of future_data for MPI
    // -----------------------------------------------------------------
    struct future_data : hpx::lcos::detail::future_data<void>
    {
        HPX_NON_COPYABLE(future_data);

        using init_no_addref = hpx::lcos::detail::future_data<void>::init_no_addref;

        // default empty constructor
        future_data() = default;

        // constructor that takes a request
        future_data(init_no_addref no_addref, MPI_Request request)
            : hpx::lcos::detail::future_data<void>(no_addref), request_(request)
        {}

        // constructor used for creation directly by invoke
        future_data(init_no_addref no_addref)
            : hpx::lcos::detail::future_data<void>(no_addref)
        {}

        // The native MPI request handle owned by this future data
        MPI_Request request_;
    };

    // -----------------------------------------------------------------
    // utility function to add a new request to the list to be tracked
    void add_to_request_list(future_data_ptr data)
    {
        // lock the MPI work list before appending new request
        std::unique_lock<std::mutex> lk(list_mtx_);
        // debugging
        // for debugging only
        if (mpi_debug.is_enabled()) {
            int rank = -1;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            mpi_debug.debug(hpx::debug::str<>("add request")
                , "R", rank
                , "request", hpx::debug::hex<8>(data->request_)
                , "list size", hpx::debug::dec<3>(active_requests_.size()+1));
        }
        // push onto list, this will make a copy and increment the ref count
        active_requests_.push_back(data);
    }

    // -----------------------------------------------------------------
    // return a future object from a user supplied MPI_Request
    hpx::future<void> get_future(MPI_Request request)
    {
        // for debugging only
        int rank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (request != MPI_REQUEST_NULL) {
            future_data_ptr data =
                new future_data(future_data::init_no_addref{}, request);

            add_to_request_list(data);

            using traits::future_access;
            return future_access<hpx::future<void>>::create(std::move(data));
        }
        return hpx::make_ready_future<void>();
    }

    // -----------------------------------------------------------------
    // return a future from an async call to MPI_Ixxx function
    template <typename F, typename ...Ts>
    hpx::future<void> async(F f, Ts &&...ts)
    {
        future_data_ptr data =
            new future_data(future_data::init_no_addref{});

        // invoke the call to MPI_Ixxx
        f(std::forward<Ts>(ts)..., &data->request_);

        // add the new shared state to the list for tracking
        add_to_request_list(data);

        // return a new future with the mpi::future_data shared state
        using traits::future_access;
        return future_access<hpx::future<void>>::create(std::move(data));
    }

    // -----------------------------------------------------------------
    // Background progress function for MPI async operations
    // Checks for completed MPI_Requests and sets mpi::future ready
    // when found
    void poll()
    {
        std::unique_lock<std::mutex> lk(list_mtx_, std::try_to_lock);
        if (!lk.owns_lock()) return;

        // for debugging
        static auto timer = mpi_debug.make_timer(1, hpx::debug::str<>("MPI_Test"));
        int rank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // list to track outstanding requests
        // list useful to allow deleting from any location
        std::list<future_data_ptr>::iterator i = active_requests_.begin();
        while (i != active_requests_.end())
        {
            MPI_Request request = (*i)->request_;
            int flag;
            MPI_Status status;
            //
            auto original_req = request;
            MPI_Test(&request, &flag, &status);
            if (flag!=0) {
                mpi_debug.debug(hpx::debug::str<>("MPI_Test set_data")
                               , "R", rank
                               , "request", hpx::debug::hex<8>(original_req)
                               , "list size", active_requests_.size());

                // mark the future as ready by setting the shared_state
                (*i)->set_data(hpx::util::unused);
                i = active_requests_.erase(i);
            }
            else {
                mpi_debug.timed(timer
                               , "no promise"
                               , "R", rank
                               , "request", hpx::debug::hex<8>(original_req)
                               , "list size", active_requests_.size());
            }
            mpi_debug.timed(timer
                           , "R", rank
                           , "request", hpx::debug::hex<8>(original_req)
                           , "list size", active_requests_.size());

            ++i;
        }
    }

    // This is not completely safe as it will return when the request list is
    // empty, but cannot guarantee that other communications are not about
    // to be launched in outstanding continuations etc.
    void wait() {
        // Yield until there is only this and background threads left.
        hpx::util::yield_while([&]() {
            return (active_requests_.size()>0);
        });
    }
}}

#endif
