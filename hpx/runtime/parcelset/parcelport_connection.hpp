//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_PARCELPORT_CONNECTION_HPP
#define HPX_PARCELSET_PARCELPORT_CONNECTION_HPP

#include <hpx/runtime/parcelset/parcel_buffer.hpp>

#include <boost/enable_shared_from_this.hpp>
#include <boost/noncopyable.hpp>

namespace hpx { namespace parcelset {

    class parcelport;
    template <typename ConnectionHandler>
    class parcelport_impl;

    boost::uint64_t get_max_inbound_size(parcelport&);

    template <typename Connection, typename BufferType, typename ChunkType = util::serialization_chunk>
    struct parcelport_connection
      : boost::enable_shared_from_this<Connection>
      , private boost::noncopyable
    {
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
        bool first_message_;
        state state_;
#endif

#if defined(HPX_HAVE_SECURITY) && !defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
        parcelport_connection()
          : first_message_(true)
        {}
        bool first_message_;
#endif

#if !defined(HPX_HAVE_SECURITY) && defined(HPX_TRACK_STATE_OF_OUTGOING_TCP_CONNECTION)
        parcelport_connection()
          : state_(state_initialized)
        {}
        state state_;
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

        virtual boost::shared_ptr<parcel_buffer_type> get_buffer(parcel const & p = parcel(), std::size_t arg_size = 0)
        {
            if(!buffer_ || (buffer_ && !buffer_->parcels_decoded_))
            {
                buffer_ = boost::make_shared<parcel_buffer_type>();
                buffer_->data_.reserve(arg_size);
            }
            return buffer_;
        }

        void reset_buffer()
        {
            buffer_.reset();
        }

        /// buffer for data
        boost::shared_ptr<parcel_buffer_type> buffer_;
    };
}}

#endif
