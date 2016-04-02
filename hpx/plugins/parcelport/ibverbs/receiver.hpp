//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_IBVERBS_RECEIVER_HPP
#define HPX_PARCELSET_POLICIES_IBVERBS_RECEIVER_HPP

#include <hpx/config/defines.hpp>
#if defined(HPX_HAVE_PARCELPORT_IBVERBS)

#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/runtime/parcelset/decode_parcels.hpp>
#include <hpx/plugins/parcelport/ibverbs/allocator.hpp>
#include <hpx/plugins/parcelport/ibverbs/context.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>

#include <boost/asio/placeholders.hpp>
#include <boost/integer/endian.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/atomic.hpp>
#include <boost/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/thread/locks.hpp>

namespace hpx { namespace parcelset { namespace policies { namespace ibverbs
{
    class connection_handler;

    ibverbs_mr register_buffer(connection_handler & handler,
        ibv_pd * pd, char * buffer, std::size_t size, int access);

    class receiver
      : public parcelport_connection<
            receiver
          , std::vector<char, allocator<message::payload_size> >
          , std::vector<char>
        >
    {
        typedef bool(receiver::*next_function_type)(boost::system::error_code &);
    public:
        /// Construct a listening parcelport_connection with the given io_service.
        receiver(connection_handler& parcelport, util::memory_chunk_pool & pool)
          : context_(), parcelport_(parcelport), memory_pool_(pool)
        {
        }

        ~receiver()
        {
            // gracefully and portably shutdown the connection
            boost::system::error_code ec;
            context_.close(ec);    // close the context
        }

        /// Get the data window associated with the parcelport_connection.
        server_context& context() { return context_; }

        boost::shared_ptr<parcel_buffer_type> get_buffer(parcel const & p = parcel(),
            std::size_t arg_size = 0)
        {
            if(!buffer_ || (buffer_ && !buffer_->parcels_decoded_))
            {
                boost::system::error_code ec;
                buffer_
                    = boost::shared_ptr<parcel_buffer_type>(
                        new parcel_buffer_type(
                            allocator<message::payload_size>(memory_pool_)
                        )
                    );
                buffer_->data_.reserve(arg_size);
            }
            return buffer_;
        }

        /// Asynchronously read a data structure from the socket.
        void async_read(boost::system::error_code & ec)
        {
            buffer_ = get_buffer();
            buffer_->clear();

            // Store the time of the begin of the read operation
            buffer_->data_point_.time_ = timer_.elapsed_nanoseconds();
            buffer_->data_point_.serialization_time_ = 0;
            buffer_->data_point_.bytes_ = 0;
            buffer_->data_point_.num_parcels_ = 0;

            context_.post_receive(ec, true);
            next(&receiver::read_size);
        }

        bool done(connection_handler & pp, boost::system::error_code & ec)
        {
            next_function_type f = 0;
            {
                boost::lock_guard<hpx::lcos::local::spinlock> l(mtx_);
                f = next_;
            }
            if(f != 0)
            {
                if(((*this).*f)(ec))
                {
                    // take measurement of overall receive time
                    buffer_->data_point_.time_ = timer_.elapsed_nanoseconds() -
                        buffer_->data_point_.time_;

                    // decode the received parcels.
                    decode_parcels(pp, *this, buffer_);
                    mr_.reset();
                    return true;
                }
            }
            return false;
        }

    private:
        bool read_size(boost::system::error_code & ec)
        {
            if(context_.check_wc<false>(MSG_SIZE, ec))
            {
                std::size_t size = context_.connection().size();
                buffer_->data_.resize(size);
                if(size <= message::payload_size)
                {
                    std::memcpy(&buffer_->data_[0], context_.connection().msg_payload(),
                        size);
                    return next(&receiver::write_ack);
                }
                else
                {
                    mr_ = register_buffer(
                        parcelport_
                      , context_.pd_
                      , &buffer_->data_[0]
                      , buffer_->data_.size()
                      , IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
                    adapted_mr_ = *mr_.mr_;
                    adapted_mr_.addr = &buffer_->data_[0];
                    adapted_mr_.length = buffer_->data_.size();

                    // write the newly received mr ...
                    context_.connection().send_mr(&adapted_mr_, ec);
                    return next(&receiver::sent_mr);
                }
            }
            return false;
        }

        bool sent_mr(boost::system::error_code & ec)
        {
            if(context_.check_wc<false>(MSG_MR, ec))
            {
                context_.post_receive(ec);
                return next(&receiver::read_data);
            }
            return false;
        }

        bool read_data(boost::system::error_code & ec)
        {
            if(context_.check_wc<false>(MSG_DATA, ec))
            {
                return write_ack(ec);
            }
            return false;
        }

        bool write_ack(boost::system::error_code & ec)
        {
            context_.connection().send_ready(ec);
            return next(&receiver::wrote_ack);
        }

        bool wrote_ack(boost::system::error_code & ec)
        {
            if(context_.check_wc<false>(MSG_DONE, ec))
            {
                return true;
            }
            return false;
        }

        bool next(next_function_type f)
        {
            boost::lock_guard<hpx::lcos::local::spinlock> l(mtx_);
            next_ = f;
            return false;
        }

        /// Data window for the receiver.
        server_context context_;
        hpx::lcos::local::spinlock mtx_;
        next_function_type next_;

        /// The handler used to process the incoming request.
        connection_handler& parcelport_;

        util::memory_chunk_pool<> & memory_pool_;
        ibverbs_mr mr_;
        ibv_mr adapted_mr_;

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

#endif
