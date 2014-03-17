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
    class sender
      : public parcelset::parcelport_connection<sender, data_buffer>
    {
    public:
        sender(boost::asio::io_service& io_service,
            naming::locality const& there,
            performance_counters::parcels::gatherer& parcels_sent)
          : context_(io_service),
            there_(there), parcels_sent_(parcels_sent)
        {
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
                boost::system::error_code ec;
                std::string buffer_size_str = get_config_entry("hpx.parcel.ibverbs.buffer_size", "4096");

                std::size_t buffer_size = boost::lexical_cast<std::size_t>(buffer_size_str);
                char * mr_buffer = context_.set_buffer_size(buffer_size, ec);

                buffer_ = boost::shared_ptr<parcel_buffer_type>(new parcel_buffer_type());
                buffer_->data_.set_mr_buffer(mr_buffer, buffer_size);
            }
            return buffer_;
        }

        /// Asynchronously write a data structure to the socket.
        template <typename Handler, typename ParcelPostprocess>
        void async_write(Handler handler, ParcelPostprocess parcel_postprocess)
        {
            /// Increment sends and begin timer.
            buffer_->data_point_.time_ = timer_.elapsed_nanoseconds();

            void (sender::*f)(boost::system::error_code const&, std::size_t,
                    boost::tuple<Handler, ParcelPostprocess>)
                = &sender::handle_write<Handler, ParcelPostprocess>;

            context_.async_write(buffer_->data_,
                boost::bind(f, shared_from_this(),
                    boost::asio::placeholders::error, ::_2,
                    boost::make_tuple(handler, parcel_postprocess)));
        }

    private:
        /// handle completed write operation
        template <typename Handler, typename ParcelPostprocess>
        void handle_write(boost::system::error_code const& e, std::size_t bytes,
            boost::tuple<Handler, ParcelPostprocess> handler)
        {
            // just call initial handler
            boost::get<0>(handler)(e, bytes);

            // complete data point and push back onto gatherer
            buffer_->data_point_.time_ =
                timer_.elapsed_nanoseconds() - buffer_->data_point_.time_;
            parcels_sent_.add_data(buffer_->data_point_);

            // now we can give this connection back to the cache
            buffer_->clear();

            buffer_->data_point_.bytes_ = 0;
            buffer_->data_point_.time_ = 0;
            buffer_->data_point_.serialization_time_ = 0;
            buffer_->data_point_.num_parcels_ = 0;

            // now handle the acknowledgement byte which is sent by the receiver
            void (sender::*f)(boost::system::error_code const&,
                      boost::tuple<Handler, ParcelPostprocess>)
                = &sender::handle_read_ack<Handler, ParcelPostprocess>;

            context_.async_read_ack(boost::bind(f, shared_from_this(),
                boost::asio::placeholders::error, handler));
        }

        template <typename Handler, typename ParcelPostprocess>
        void handle_read_ack(boost::system::error_code const& e,
            boost::tuple<Handler, ParcelPostprocess> handler)
        {
            // Call post-processing handler, which will send remaining pending
            // parcels. Pass along the connection so it can be reused if more
            // parcels have to be sent.
            boost::get<1>(handler)(e, there_, shared_from_this());
        }

        /// Context for the parcelport_connection.
        client_context context_;

        /// the other (receiving) end of this connection
        naming::locality there_;
        /// Counters and their data containers.
        util::high_resolution_timer timer_;
        performance_counters::parcels::gatherer& parcels_sent_;
    };
}}}}

#endif
