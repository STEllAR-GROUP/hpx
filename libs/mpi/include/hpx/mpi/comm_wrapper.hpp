//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_MPI_COMM_WRAPPER_HPP
#define HPX_MPI_COMM_WRAPPER_HPP

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
//
#include <hpx/local_lcos/promise.hpp>
#include <hpx/local_lcos/channel.hpp>
//
#include <hpx/debugging/print.hpp>
#include <hpx/mpi/mpi_future.hpp>

namespace hpx { namespace mpi {

    using  print_on = hpx::debug::enable_print<false>;
    static print_on com_debug("MPI_COM");

    template <typename Comm>
    struct comm_wrapper {
        //
        comm_wrapper(const Comm &comm) : comm_(comm)
        {
//            com_debug.debug(hpx::debug::str<>("construct comm_wrapper"));
            channel_counter_ = 0;
            //
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            rank_ = rank;
        }

        // storing wrappers in a vector requires move operator (resizing etc)
        comm_wrapper(comm_wrapper<Comm> &&other)
            : comm_(std::move(other.comm_))
            , rank_(std::move(other.rank_))
            , channel_(std::move(other.channel_))
        {
//            com_debug.debug(hpx::debug::str<>("move comm_wrapper"));
            channel_counter_.store(other.channel_counter_);
        }

        ~comm_wrapper()
        {
            mpi_debug.debug(hpx::debug::str<>("~comm_wrapper"));
        }

        Comm &get_comm() { return comm_; }

        Comm *operator->() { return &comm_; }

        void make_ready() {
            triggerChannel();
        }

        hpx::future<comm_wrapper<Comm>*> get_future()
        {
            return channel_.get();
        }

        int triggerChannel()
        {
            channel_.set(this);
            return --channel_counter_;
        }

        int channelCounter(bool inc)
        {
            if (inc) ++channel_counter_;
            return channel_counter_;
        }

        Comm comm_;
        // For experimental hpx/mpi work
        //mutable hpx::lcos::local::promise<const comm_wrapper2D*> promise_;
        mutable hpx::lcos::local::channel<comm_wrapper<Comm>*> channel_;
        mutable std::atomic<int> channel_counter_;
        int rank_;
    };

    template <typename Comm>
    using comm_vector_type = std::vector<comm_wrapper<Comm>>;

    template <typename Comm>
    using comm_reference = comm_wrapper<Comm>*;

    template <typename Comm>
    using comm_future = hpx::future<comm_reference<Comm>>;

    template <typename Comm>
    comm_vector_type<Comm> make_communicator_array(Comm const &original, unsigned N)
    {
        com_debug.debug(hpx::debug::str<>("make_communicator_array"), N
            , "R", hpx::mpi::detail::mpi_info_.rank_);
        comm_vector_type<Comm> result;
        result.reserve(N);
        for (unsigned i=0; i<N; ++i) {
            // construct a wrapped copy of the communicator on our array
            result.emplace_back(original);
            // make the first one ready
            result.back().make_ready();
        }
        return result;
    }

    template <typename Comm>
    void delete_communicator_array(comm_vector_type<Comm> &commarray)
    {
        mpi_debug.debug(hpx::debug::str<>("delete_communicator_array"), commarray.size()
            , "R", hpx::mpi::detail::mpi_info_.rank_);
        for (auto &c : commarray) {
            // actions?
        }
        // just empty the array
        commarray = {};
    }

    template <typename Comm>
    comm_future<Comm> get_communicator(unsigned index, comm_vector_type<Comm> &comms)
    {
        auto &c = comms[index];
        return c.get_future();
    }

    template <typename Comm>
    comm_future<Comm> get_communicator_with_debug(unsigned index, comm_vector_type<Comm> &comms, const char *info, unsigned v)
    {
        auto &c = comms[index];
        com_debug.debug(hpx::debug::str<>("get_communicator")
                        , info
                        , "R", hpx::mpi::detail::mpi_info_.rank_
                        , "Ix", index
                        , "V", v
                        );
//                        , "Ct", hpx::debug::dec<3>(RowCommunicators[index].channelCounter(true)));
        return c.get_future();
    }

}}

#endif
