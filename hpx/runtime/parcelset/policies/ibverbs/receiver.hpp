//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

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
      : public parcelport_connection<receiver, data_buffer, std::vector<char> >
    {
    public:
        /// Construct a listening parcelport_connection with the given io_service.
        receiver(boost::asio::io_service& io_service,
                connection_handler& parcelport)
          : context_(io_service),
            parcelport_(parcelport)
        {
        }

        ~receiver()
        {
            // gracefully and portably shutdown the connection
            boost::system::error_code ec;
            context_.close(ec);    // close the context
        }

        boost::shared_ptr<parcel_buffer_type> get_buffer(parcel const & p = parcel(), std::size_t arg_size = 0)
        {
            if(!buffer_)
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

        /// Get the data window associated with the parcelport_connection.
        server_context& context() { return context_; }
        
        /// Asynchronously read a data structure from the socket.
        template <typename Handler>
        void async_read(Handler handler)
        {
            buffer_ = get_buffer();

            // Store the time of the begin of the read operation
            buffer_->data_point_.time_ = timer_.elapsed_nanoseconds();
            buffer_->data_point_.serialization_time_ = 0;
            buffer_->data_point_.bytes_ = 0;
            buffer_->data_point_.num_parcels_ = 0;

            // Issue a read operation to read the parcel priority and size.
            void (receiver::*f)(boost::system::error_code const&,
                    boost::tuple<Handler>)
                = &receiver::handle_read_data<Handler>;

            context_.async_read(buffer_->data_,
                boost::bind(f, shared_from_this(),
                    boost::asio::placeholders::error,
                    boost::make_tuple(handler)));
        }

    private:
        /// Handle a completed read of message data.
        template <typename Handler>
        void handle_read_data(boost::system::error_code const& e,
            boost::tuple<Handler> handler)
        {
            if (e) {
                boost::get<0>(handler)(e);

                // Issue a read operation to read the next parcel.
                async_read(boost::get<0>(handler));
            }
            else {
                // complete data point and pass it along
                buffer_->data_point_.time_ = timer_.elapsed_nanoseconds() -
                    buffer_->data_point_.time_;

                // now send acknowledgment message
                void (receiver::*f)(boost::system::error_code const&, 
                          boost::tuple<Handler>)
                    = &receiver::handle_write_ack<Handler>;

                // decode the received parcels.
                decode_parcels(parcelport_, shared_from_this(), buffer_);

                // acknowledge to have received the parcel
                context_.async_write_ack(
                    boost::bind(f, shared_from_this(), 
                        boost::asio::placeholders::error, handler));
            }
        }

        template <typename Handler>
        void handle_write_ack(boost::system::error_code const& e,
            boost::tuple<Handler> handler)
        {
            // Inform caller that data has been received ok.
            boost::get<0>(handler)(e);

            // Issue a read operation to handle the next parcel.
            async_read(boost::get<0>(handler));
        }

        /// Data window for the receiver.
        server_context context_;

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
