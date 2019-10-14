//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_PARCELPORT_CONNECTION_HPP
#define HPX_PARCELSET_PARCELPORT_CONNECTION_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/assertion.hpp>
#include <hpx/runtime/parcelset/parcel_buffer.hpp>

#include <cstdint>
#include <memory>
#include <utility>

namespace hpx { namespace parcelset {

    class parcelport;
    template <typename ConnectionHandler>
    class parcelport_impl;

    std::int64_t HPX_EXPORT get_max_inbound_size(parcelport&);

    template <typename Connection, typename BufferType,
        typename ChunkType = serialization::serialization_chunk>
    struct parcelport_connection
      : std::enable_shared_from_this<Connection>
    {
          ////////////////////////////////////////////////////////////////////////
          typedef BufferType buffer_type;
          typedef parcel_buffer<buffer_type, ChunkType> parcel_buffer_type;

    public:
        HPX_NON_COPYABLE(parcelport_connection);

    public:
#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
        enum state
        {
            state_initialized,
            state_reinitialized,
            state_set_parcel,
            state_async_write,
            state_handle_write,
            state_handle_read_ack,
            state_scheduled_thread,
            state_send_pending,
            state_reclaimed,
            state_deleting
        };
#endif

#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
        parcelport_connection()
          : state_(state_initialized)
        {}

        parcelport_connection(typename BufferType::allocator_type const & alloc)
          : state_(state_initialized)
          , buffer_(alloc)
        {}
        state state_;
#else
        parcelport_connection()
        {}

        parcelport_connection(typename BufferType::allocator_type const & alloc)
          : buffer_(alloc)
        {}

        parcelport_connection(typename BufferType::allocator_type * alloc)
          : buffer_(std::move(buffer_type(alloc)),alloc)
        {}

        parcelport_connection(parcel_buffer_type && buffer)
          : buffer_(std::move(buffer))
        {}

#endif

#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
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
#endif

        virtual ~parcelport_connection() {}

        /// buffer for data
        parcel_buffer_type buffer_;
    };
}}

#endif
#endif
