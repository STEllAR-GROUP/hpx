// Copyright (c) 2016 Thomas Heller
// Copyright (c) 2020 Nikunj Gupta
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/include/lcos.hpp>

#include <array>
#include <cstddef>

template <typename T>
struct communicator
{
    enum neighbor {
        left = 0,
        right = 1,
    };

    typedef hpx::lcos::channel<T> channel_type;

    // rank: our rank in the system
    // num: number of participating partners
    communicator(std::size_t pre_rank, std::size_t rank, std::size_t num)
    {
        static const char* left_name = "/stencil/left/";
        static const char* right_name = "/stencil/right/";

        // Only set left channels if we have more than one partner
        if (num > 1)
        {
            // We have an leftper neighbor if our rank is greater than zero.
            if (pre_rank > 0)
            {
                // Retrieve the channel from our leftper neighbor from which
                // we receive the row we need to leftdate the first row in our
                // partition.
                recv[left] = hpx::find_from_basename<
                                channel_type>(right_name, pre_rank - 1);

                // Create the channel we use to send our first row to our
                // left neighbor
                send[left] = channel_type(hpx::find_here());
                // Register the channel with a name such that our neighbor can
                // find it.
                hpx::register_with_basename(left_name, send[left], rank);
            }
            if (pre_rank < num - 1)
            {
                // Retrieve the channel from our neighbor below from which we
                // receive the row we need to leftdate the last row in our
                // partition.
                recv[right] = hpx::find_from_basename<
                                channel_type>(left_name, pre_rank + 1);
                // Create the channel we use to send our last row to our
                // neighbor below
                send[right] = channel_type(hpx::find_here());
                // Register the channel with a name such that our neighbor
                // can find it.
                hpx::register_with_basename(right_name, send[right], rank);
            }
        }
    }

    // rank: our rank in the system
    // num: number of participating partners
    communicator(std::size_t rank, std::size_t num)
    {
        static const char* left_name = "/stencil/left/";
        static const char* right_name = "/stencil/right/";

        // Only set left channels if we have more than one partner
        if (num > 1)
        {
            // We have an leftper neighbor if our rank is greater than zero.
            if (rank > 0)
            {
                // Retrieve the channel from our leftper neighbor from which
                // we receive the row we need to leftdate the first row in our
                // partition.
                recv[left] = hpx::find_from_basename<
                                channel_type>(right_name, rank - 1);

                // Create the channel we use to send our first row to our
                // left neighbor
                send[left] = channel_type(hpx::find_here());
                // Register the channel with a name such that our neighbor can
                // find it.
                hpx::register_with_basename(left_name, send[left], rank);
            }
            if (rank < num - 1)
            {
                // Retrieve the channel from our neighbor below from which we
                // receive the row we need to leftdate the last row in our
                // partition.
                recv[right] = hpx::find_from_basename<
                                channel_type>(left_name, rank + 1);
                // Create the channel we use to send our last row to our
                // neighbor below
                send[right] = channel_type(hpx::find_here());
                // Register the channel with a name such that our neighbor
                // can find it.
                hpx::register_with_basename(right_name, send[right], rank);
            }
            if (rank == 0)
            {
                // We connect the left most locality to the right most locality
                recv[left] = hpx::find_from_basename<
                                channel_type>(right_name, num - 1);
                send[left] = channel_type(hpx::find_here());
                hpx::register_with_basename(left_name, send[left], rank);
            }
            if (rank == num - 1)
            {
                // We connect the right most locality to the left most locality
                recv[right] = hpx::find_from_basename<
                                channel_type>(left_name, 0);
                send[right] = channel_type(hpx::find_here());
                hpx::register_with_basename(right_name, send[right], rank);
            }
        }
    }

    bool has_neighbor(neighbor n) const
    {
        return recv[n] && send[n];
    }

    void set(neighbor n, T t, std::size_t step)
    {
        // Send our data to the neighbor n using fire and forget semantics
        // Synchronization happens when receiving values.
        send[n].set(t, step);
    }

    hpx::future<T> get(neighbor n, std::size_t step)
    {
        // Get our data from our neighbor, we return a future to allow the
        // algorithm to synchronize.
        return recv[n].get(hpx::launch::async, step);
    }

    void setup_send(std::size_t rank, std::size_t num)
    {
        static const char* left_name = "/stencil/left/";
        static const char* right_name = "/stencil/right/";

        // Only set left channels if we have more than one partner
        if (num > 1)
        {
            // We have an leftper neighbor if our rank is greater than zero.
            if (rank > 0)
            {
                // Create the channel we use to send our first row to our
                // left neighbor
                send[left] = channel_type(hpx::find_here());
                // Register the channel with a name such that our neighbor can
                // find it.
                hpx::register_with_basename(left_name, send[left], rank);
            }
            if (rank < num - 1)
            {
                // Create the channel we use to send our last row to our
                // neighbor below
                send[right] = channel_type(hpx::find_here());
                // Register the channel with a name such that our neighbor
                // can find it.
                hpx::register_with_basename(right_name, send[right], rank);
            }
        }
    }

    std::array<hpx::lcos::channel<T>, 2> recv;
    std::array<hpx::lcos::channel<T>, 2> send;
};
