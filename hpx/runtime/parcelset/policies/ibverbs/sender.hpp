//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_IBVERBS_SENDER_HPP
#define HPX_PARCELSET_POLICIES_IBVERBS_SENDER_HPP

#include <hpx/runtime/naming/locality.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/context.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/messages.hpp>
#include <hpx/runtime/parcelset/policies/ibverbs/data_buffer.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <boost/asio/placeholders.hpp>

namespace hpx { namespace parcelset { namespace policies { namespace ibverbs
{
    class connection_handler;
    void add_sender(connection_handler & handler,
        boost::shared_ptr<sender> const& sender_connection);

    class sender
      : public parcelset::parcelport_connection<sender, data_buffer>
    {
        typedef bool(sender::*next_function_type)();
    public:
        typedef
            HPX_STD_FUNCTION<void(boost::system::error_code const &, std::size_t)>
            handler_function_type;
        typedef
            HPX_STD_FUNCTION<
                void(
                    boost::system::error_code const &
                  , naming::locality const&
                  ,  boost::shared_ptr<sender>
                )
            >
            postprocess_function_type;

        sender(connection_handler & handler, naming::locality const& there,
            performance_counters::parcels::gatherer& parcels_sent)
          : parcelport_(handler), there_(there), parcels_sent_(parcels_sent)
        {
            boost::system::error_code ec;
            std::string buffer_size_str = get_config_entry("hpx.parcel.ibverbs.buffer_size", "4096");

            buffer_size_ = boost::lexical_cast<std::size_t>(buffer_size_str);
            mr_buffer_ = context_.set_buffer_size(buffer_size_, ec);
        }

        ~sender()
        {
            // gracefully and portably shutdown the socket
            boost::system::error_code ec;
            context_.shutdown(ec); // shut down data connection
            context_.close(ec);    // close the socket to give it back to the OS
        }

        /// Get the window associated with the parcelport_connection.
        client_context& context() { return context_; }

        void verify(naming::locality const & parcel_locality_id)
        {
            HPX_ASSERT(parcel_locality_id == there_);
        }

        naming::locality const& destination() const
        {
            return there_;
        }

        boost::shared_ptr<parcel_buffer_type> get_buffer(parcel const & p, std::size_t arg_size)
        {
            if(!buffer_ || (buffer_ && !buffer_->parcels_decoded_))
            {
                buffer_ = boost::shared_ptr<parcel_buffer_type>(new parcel_buffer_type());
                buffer_->data_.set_mr_buffer(mr_buffer_, buffer_size_);
            }
            return buffer_;
        }

        /// Asynchronously write a data structure to the socket.
        template <typename Handler, typename ParcelPostprocess>
        void async_write(Handler handler, ParcelPostprocess parcel_postprocess)
        {
            /// Increment sends and begin timer.
            buffer_->data_point_.time_ = timer_.elapsed_nanoseconds();

            handler_ = handler;
            postprocess_ = parcel_postprocess;
            
            next(&sender::send_data);
            add_sender(parcelport_, shared_from_this());
        }
        
        bool done()
        {
            next_function_type f = 0;
            {
                hpx::lcos::local::spinlock::scoped_lock l(mtx_);
                f = next_;
            }
            if(f != 0)
            {
                if(((*this).*f)())
                {
                    error_code ec;
                    handler_(ec, buffer_->data_.size());
                    buffer_->data_point_.time_ = timer_.elapsed_nanoseconds()
                        - buffer_->data_point_.time_;
                    parcels_sent_.add_data(buffer_->data_point_);
                    postprocess_function_type pp;
                    std::swap(pp, postprocess_);
                    pp(ec, there_, shared_from_this());
                    return true;
                }
            }
            return false;
        }

    private:
        bool send_data()
        {
            context_.write(buffer_->data_, boost::system::throws);
            return next(&sender::read_ack);
        }

        bool read_ack()
        {
            if(context_.try_read_ack(boost::system::throws))
            {
                next(0);
                return true;
            }
            return next(&sender::read_ack);
        }

        bool next(next_function_type f)
        {
            hpx::lcos::local::spinlock::scoped_lock l(mtx_);
            next_ = f;
            return false;
        }

        /// Context for the parcelport_connection.
        client_context context_;
        std::size_t buffer_size_;
        char * mr_buffer_;
        
        hpx::lcos::local::spinlock mtx_;
        next_function_type next_;
        
        handler_function_type handler_;
        postprocess_function_type postprocess_;
        
        connection_handler & parcelport_;

        /// the other (receiving) end of this connection
        naming::locality there_;
        /// Counters and their data containers.
        util::high_resolution_timer timer_;
        performance_counters::parcels::gatherer& parcels_sent_;
    };
}}}}

#endif
