//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_IBVERBS_RECEIVER_HPP
#define HPX_PARCELSET_POLICIES_IBVERBS_RECEIVER_HPP


#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/runtime/parcelset/decode_parcels.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/data_buffer.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/context.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>

#include <boost/asio/placeholders.hpp>
#include <boost/integer/endian.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/atomic.hpp>
#include <boost/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>

namespace hpx { namespace parcelset { namespace policies { namespace ibverbs
{
    class connection_handler;

    class receiver
      : public parcelport_connection<receiver, data_buffer, util::managed_chunk>
    {
        typedef bool(receiver::*next_function_type)();
    public:
        /// Construct a listening parcelport_connection with the given io_service.
        receiver(connection_handler& parcelport)
          : parcelport_(parcelport)
        {
            boost::system::error_code ec;
            std::string buffer_size_str = get_config_entry("hpx.parcel.ibverbs.buffer_size", "4096");

            buffer_size_ = boost::lexical_cast<std::size_t>(buffer_size_str);
            mr_buffer_ = context_.set_buffer_size(buffer_size_, ec);
        }

        ~receiver()
        {
            // gracefully and portably shutdown the connection
            boost::system::error_code ec;
            context_.close(ec);    // close the context
        }

        boost::shared_ptr<parcel_buffer_type> get_buffer(parcel const & p = parcel(), std::size_t arg_size = 0)
        {
            if(!buffer_ || (buffer_ && !buffer_->parcels_decoded_))
            {
                buffer_ = boost::shared_ptr<parcel_buffer_type>(new parcel_buffer_type());
                buffer_->data_.set_mr_buffer(mr_buffer_, buffer_size_);
            }
            return buffer_;
        }

        /// Get the data window associated with the parcelport_connection.
        server_context& context() { return context_; }

        /// Asynchronously read a data structure from the socket.
        void async_read()
        {
            buffer_ = get_buffer();
            buffer_->clear();

            // Store the time of the begin of the read operation
            buffer_->data_point_.time_ = timer_.elapsed_nanoseconds();
            buffer_->data_point_.serialization_time_ = 0;
            buffer_->data_point_.bytes_ = 0;
            buffer_->data_point_.num_parcels_ = 0;

            next(&receiver::read_data);
        }
        
        bool done(connection_handler & pp)
        {
            HPX_ASSERT(next_ != 0);
            if(((*this).*next_)())
            {
                // take measurement of overall receive time
                buffer_->data_point_.time_ = timer_.elapsed_nanoseconds() -
                    buffer_->data_point_.time_;

                // decode the received parcels.
                decode_parcels(pp, shared_from_this(), buffer_);
                return true;
            }
            return false;
        }

    private:
        bool read_data()
        {
            std::size_t size = context_.try_read_data(buffer_->data_, boost::system::throws);
            if(size == 0)
            {
                return next(&receiver::read_data);
            }
            return next(&receiver::write_ack);
        }

        bool write_ack()
        {
            context_.write_ack(boost::system::throws);
            next(0);
            return true;
        }

        bool next(next_function_type f)
        {
            next_ = f;
            return false;
        }

        /// Data window for the receiver.
        server_context context_;
        std::size_t buffer_size_;
        char * mr_buffer_;
        next_function_type next_;

        /// The handler used to process the incoming request.
        connection_handler& parcelport_;

        /// Counters and timers for parcels received.
        util::high_resolution_timer timer_;
    };

    // this makes sure we can store our connections in a set
    inline bool operator<(boost::shared_ptr<receiver> const& lhs,
        boost::shared_ptr<receiver> const& rhs)
    {
        return lhs.get() < rhs.get();
    }
}}}}

#endif
