//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_PARCELPORT_CONNECTION_HPP
#define HPX_PARCELSET_PARCELPORT_CONNECTION_HPP

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
    private:
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

#if defined(HPX_HAVE_SECURITY) && defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
        parcelport_connection()
          : first_message_(true)
          , state_(state_initialized)
        {}

        parcelport_connection(typename BufferType::allocator_type const & alloc)
          : first_message_(true)
          , state_(state_initialized)
          , buffer_(alloc)
        {}
        bool first_message_;
        state state_;
#endif

#if defined(HPX_HAVE_SECURITY) && !defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
        parcelport_connection()
          : first_message_(true)
        {}
        bool first_message_;

        parcelport_connection(typename BufferType::allocator_type const & alloc)
          : first_message_(true)
          , buffer_(alloc)
        {}
        bool first_message_;
#endif

#if !defined(HPX_HAVE_SECURITY) && defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
        parcelport_connection()
          : state_(state_initialized)
        {}

        parcelport_connection(typename BufferType::allocator_type const & alloc)
          : state_(state_initialized)
          , buffer_(alloc)
        {}
        state state_;
#endif

#if !defined(HPX_HAVE_SECURITY) && !defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
        parcelport_connection()
        {}

        parcelport_connection(typename BufferType::allocator_type const & alloc)
          : buffer_(alloc)
        {}

        parcelport_connection(typename BufferType::allocator_type * alloc)
          : buffer_(std::move(buffer_type(alloc)),alloc)
        {}
#endif

#if defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
        void set_state(state newstate)
        {
            state_ = newstate;
        }
#endif

        virtual ~parcelport_connection() {}

        ////////////////////////////////////////////////////////////////////////
        typedef BufferType buffer_type;
        typedef parcel_buffer<buffer_type, ChunkType> parcel_buffer_type;

        /// buffer for data
        parcel_buffer_type buffer_;
    };
}}

#endif
