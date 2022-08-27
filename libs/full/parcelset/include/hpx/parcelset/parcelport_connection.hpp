//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/assert.hpp>
#include <hpx/modules/serialization.hpp>

#include <hpx/parcelset/parcel_buffer.hpp>
#include <hpx/parcelset/parcelset_fwd.hpp>

#include <cstdint>
#include <memory>
#include <utility>

namespace hpx::parcelset {

    template <typename Connection, typename BufferType,
        typename ChunkType = serialization::serialization_chunk>
    struct parcelport_connection : std::enable_shared_from_this<Connection>
    {
        using buffer_type = BufferType;
        using parcel_buffer_type = parcel_buffer<buffer_type, ChunkType>;

        parcelport_connection(parcelport_connection const&) = delete;
        parcelport_connection(parcelport_connection&&) = delete;
        parcelport_connection& operator=(parcelport_connection const&) = delete;
        parcelport_connection& operator=(parcelport_connection&&) = delete;

    public:
#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
        enum state{state_initialized, state_reinitialized, state_set_parcel,
            state_async_write, state_handle_write, state_handle_read_ack,
            state_scheduled_thread, state_send_pending, state_reclaimed,
            state_deleting};

        parcelport_connection()
          : state_(state_initialized)
        {
        }

        explicit parcelport_connection(
            typename BufferType::allocator_type const& alloc)
          : state_(state_initialized)
          , buffer_(alloc)
        {
        }

        void set_state(state newstate)
        {
            if (newstate == state_send_pending)
            {
                HPX_ASSERT(state_ == state_initialized ||
                    state_ == state_reinitialized ||
                    state_ == state_handle_read_ack);
            }
            state_ = newstate;
        }

        state state_;
#else
        parcelport_connection() = default;
        explicit parcelport_connection(
            typename BufferType::allocator_type const& alloc)
          : buffer_(alloc)
        {
        }

        explicit parcelport_connection(parcel_buffer_type&& buffer) noexcept
          : buffer_(HPX_MOVE(buffer))
        {
        }
#endif
        virtual ~parcelport_connection() = default;

        parcel_buffer_type buffer_;    // buffer for data
    };
}    // namespace hpx::parcelset

#endif
