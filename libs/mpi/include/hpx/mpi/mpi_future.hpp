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
#include <hpx/memory/intrusive_ptr.hpp>
#include <hpx/debugging/print.hpp>
//
//#define HPX_MPI_LIST_STORAGE
#define HPX_MPI_VECTOR_STORAGE

namespace hpx { namespace mpi {

    using  print_on = hpx::debug::enable_print<false>;
    static print_on mpi_debug("MPI_FUTURE");

    namespace detail {
        // mutex needed to protect mpi request list
        static std::mutex list_mtx_;

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

        struct mpi_info {
            bool initialized_;
            int  rank_;
            int  size_;
        };

        // we track requests and future data in two vectors even though
        // we have the request stored in the future data already
        // the reason for this is becaue we can use MPI_Testany
        // with a vector of requests to save overheads compared
        // to testing one by one every item in our list/vector
        using  future_data_ptr = memory::intrusive_ptr<future_data>;
#ifdef HPX_MPI_LIST_STORAGE
        static std::list<future_data_ptr>   active_futures_;
#else
        static std::vector<MPI_Request>     active_requests_;
        static std::vector<future_data_ptr> active_futures_;
#endif
        static mpi_info mpi_info_ = {false, 0, 0};

        // -----------------------------------------------------------------
        // utility function to add a new request to the list to be tracked
        void add_to_request_list(future_data_ptr data)
        {
            if (mpi_debug.is_enabled() && !mpi_info_.initialized_) {
                mpi_info_.initialized_ = true;
                MPI_Comm_rank(MPI_COMM_WORLD, &mpi_info_.rank_);
                MPI_Comm_size(MPI_COMM_WORLD, &mpi_info_.size_);
            }
            // lock the MPI work list before appending new request
            std::unique_lock<std::mutex> lk(list_mtx_);

            // this will make a copy and increment the ref count
            active_futures_.push_back(data);
#ifdef HPX_MPI_VECTOR_STORAGE
            active_requests_.push_back(data->request_);
            mpi_debug.debug(hpx::debug::str<>("request ptr")
                , hpx::debug::ptr(active_requests_.data()));
#endif
            // for debugging only
            if (mpi_debug.is_enabled()) {
                mpi_debug.debug(hpx::debug::str<>("add request")
                    , "R", mpi_info_.rank_
                    , "request", hpx::debug::hex<8>(data->request_)
                    , "list size", hpx::debug::dec<3>(active_futures_.size()));
            }
        }
    }

    // -----------------------------------------------------------------
    // return a future object from a user supplied MPI_Request
    hpx::future<void> get_future(MPI_Request request)
    {
        // for debugging only
        int rank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (request != MPI_REQUEST_NULL) {
            detail::future_data_ptr data =
                new detail::future_data(detail::future_data::init_no_addref{}, request);

            detail::add_to_request_list(data);

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
        detail::future_data_ptr data =
            new detail::future_data(detail::future_data::init_no_addref{});

        // invoke the call to MPI_Ixxx
        f(std::forward<Ts>(ts)..., &data->request_);

        // add the new shared state to the list for tracking
        detail::add_to_request_list(data);

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
        std::unique_lock<std::mutex> lk(detail::list_mtx_, std::try_to_lock);
        if (!lk.owns_lock()) return;

        // for debugging
        static auto timer = mpi_debug.make_timer(1, hpx::debug::str<>("MPI_Test"));

#ifdef HPX_MPI_LIST_STORAGE
        std::list<detail::future_data_ptr>::iterator i = detail::active_futures_.begin();
         while (i != detail::active_futures_.end())
        {
            MPI_Request request = (*i)->request_;
            int flag;
            MPI_Status status;
            //
            auto req = request;
            MPI_Test(&request, &flag, &status);
            if (flag!=0) {
                mpi_debug.debug(hpx::debug::str<>("MPI_Test set_data")
                                , "R", hpx::debug::dec<>(detail::mpi_info_.rank_)
                                , "S", hpx::debug::dec<>(detail::mpi_info_.size_)
                               , "request", hpx::debug::hex<8>(req)
                               , "list size", detail::active_futures_.size());

                // mark the future as ready by setting the shared_state
                (*i)->set_data(hpx::util::unused);
                i = detail::active_futures_.erase(i);
            }
            else {
                mpi_debug.timed(timer
                               , "no promise"
                               , "request", hpx::debug::hex<8>(req)
                               , "list size", detail::active_futures_.size());
            }
            mpi_debug.timed(timer
                           , "request", hpx::debug::hex<8>(req)
                           , "list size", detail::active_futures_.size());

            ++i;
        }
#else
        bool keep_trying = detail::active_requests_.size()>0;
        while (keep_trying) {
            int        index = 0;
            int        flag  = false;
            MPI_Status status;
            int result = MPI_Testany(detail::active_requests_.size(),
                            &detail::active_requests_[0],
                            &index,
                            &flag,
                            &status);
            static auto poll_deb = mpi_debug.make_timer(1
                , "R", hpx::debug::dec<>(detail::mpi_info_.rank_)
                , "S", hpx::debug::dec<>(detail::mpi_info_.size_));

            if (result == MPI_SUCCESS && index!=MPI_UNDEFINED) {
                mpi_debug.debug(poll_deb
                    , hpx::debug::str<>("Success")
                    , "index", hpx::debug::dec<>(index)
                    , "flag", hpx::debug::dec<>(flag)
                    , "status", hpx::debug::dec<>(status.MPI_ERROR));
                keep_trying = flag;
                if (keep_trying) {
                    auto req = detail::active_requests_[unsigned(index)];
                    mpi_debug.debug(hpx::debug::str<>("MPI_Testany set_data")
                                   , "request", hpx::debug::hex<8>(req)
                                   , "list size", detail::active_futures_.size());

                    // mark the future as ready by setting the shared_state
                    detail::active_futures_[unsigned(index)]->set_data(hpx::util::unused);
                    // remove the request from our list to prevent retesting
                    detail::active_requests_[unsigned(index)] = MPI_REQUEST_NULL;
                    detail::active_futures_[unsigned(index)]  = nullptr;
                }
            }
            else {
                keep_trying = false;
                mpi_debug.timed(poll_deb
                    , hpx::debug::str<>("Poll")
                    , "index", hpx::debug::dec<>(index)
                    , "flag", hpx::debug::dec<>(flag)
                    , "status", hpx::debug::dec<>(status.MPI_ERROR));
            }
        };
        // if there are more than 25% NULL request handles in our lists, compact them
        unsigned nulls = std::count(detail::active_requests_.begin(), detail::active_requests_.end(), MPI_REQUEST_NULL);
        if (nulls>=detail::active_requests_.size()/4) {
            auto end1 = std::remove(detail::active_requests_.begin(), detail::active_requests_.end(), MPI_REQUEST_NULL);
            detail::active_requests_.resize(std::distance(detail::active_requests_.begin(),end1));
            auto end2 = std::remove(detail::active_futures_.begin(), detail::active_futures_.end(), nullptr);
            detail::active_futures_.resize(std::distance(detail::active_futures_.begin(),end2));
        }
#endif
    }

    // This is not completely safe as it will return when the request list is
    // empty, but cannot guarantee that other communications are not about
    // to be launched in outstanding continuations etc.
    void wait() {
        hpx::util::yield_while([&]() {
            return (detail::active_futures_.size()>0);
        });
    }
}}

#endif
