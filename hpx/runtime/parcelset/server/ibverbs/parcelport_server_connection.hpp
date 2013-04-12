//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_SERVER_IBVERBS_PARCELPORT_CONNECTION_HPP)
#define HPX_PARCELSET_SERVER_IBVERBS_PARCELPORT_CONNECTION_HPP

#include <sstream>
#include <vector>

#include <hpx/runtime/parcelset/server/parcelport_queue.hpp>
#include <hpx/runtime/parcelset/ibverbs/context.hpp>
#include <hpx/runtime/parcelset/ibverbs/messages.hpp>
#include <hpx/runtime/parcelset/ibverbs/data_buffer.hpp>
#include <hpx/util/high_resolution_timer.hpp>
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

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace ibverbs
{
    // forward declaration only
    class parcelport;


    void decode_message(parcelport&,
        data_buffer const & buffer,
        boost::uint64_t inbound_data_size,
        performance_counters::parcels::data_point receive_data);
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace server { namespace ibverbs
{
    /// Represents a single parcelport_connection from a client.
    class parcelport_connection
      : public boost::enable_shared_from_this<parcelport_connection>,
        private boost::noncopyable
    {
    public:
        /// Construct a listening parcelport_connection with the given io_service.
        parcelport_connection(boost::asio::io_service& io_service,
                parcelset::ibverbs::parcelport& parcelport)
          : context_(io_service),
            parcelport_(parcelport)
        {
            boost::system::error_code ec;
            std::string buffer_size_str = get_config_entry("hpx.parcel.ibverbs.buffer_size", "4096");
            std::size_t buffer_size_ = boost::lexical_cast<std::size_t>(buffer_size_str);
            char * mr_buffer_ = context_.set_buffer_size(buffer_size_, ec);

            in_buffer_.set_mr_buffer(mr_buffer_, buffer_size_);
        }

        ~parcelport_connection()
        {
            // gracefully and portably shutdown the connection
            boost::system::error_code ec;
            context_.close(ec);    // close the context
        }

        /// Get the data window associated with the parcelport_connection.
        parcelset::ibverbs::server_context& context() { return context_; }

        /// Asynchronously read a data structure from the socket.
        template <typename Handler>
        void async_read(Handler handler)
        {
            // Store the time of the begin of the read operation
            receive_data_.time_ = timer_.elapsed_nanoseconds();
            receive_data_.serialization_time_ = 0;
            receive_data_.bytes_ = 0;
            receive_data_.num_parcels_ = 0;

            // Issue a read operation to read the parcel priority and size.
            void (parcelport_connection::*f)(boost::system::error_code const&,
                    boost::tuple<Handler>)
                = &parcelport_connection::handle_read_data<Handler>;

                
            in_buffer_.clear();

            context_.async_read(in_buffer_,
                boost::bind(f, shared_from_this(),
                    boost::asio::placeholders::error,
                    boost::make_tuple(handler)));
        }

    protected:
        /// Handle a completed read of message data.
        /// message header.
        /// The handler is passed using a tuple since boost::bind seems to have
        /// trouble binding a function object created using boost::bind as a
        /// parameter.
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
                receive_data_.time_ = timer_.elapsed_nanoseconds() -
                    receive_data_.time_;

                // now send acknowledgment message
                void (parcelport_connection::*f)(boost::system::error_code const&, 
                          boost::tuple<Handler>)
                    = &parcelport_connection::handle_write_ack<Handler>;


                // add parcel data to incoming parcel queue
                performance_counters::parcels::data_point receive_data = 
                    receive_data_;
                
                decode_message(parcelport_, in_buffer_, in_buffer_.size(), receive_data_);

                context_.async_write_ack(
                    boost::bind(f, shared_from_this(), 
                        boost::asio::placeholders::error, handler));

                // Inform caller that data has been received ok.
                boost::get<0>(handler)(e);
            }
        }

        template <typename Handler>
        void handle_write_ack(boost::system::error_code const& e,
            boost::tuple<Handler> handler)
        {
            // Issue a read operation to read the next parcel.
            async_read(boost::get<0>(handler));
        }

    private:
        /// Data window for the parcelport_connection.
        parcelset::ibverbs::server_context context_;

        /// buffer for incoming data
        parcelset::ibverbs::data_buffer in_buffer_;

        /// The handler used to process the incoming request.
        parcelset::ibverbs::parcelport& parcelport_;
        
        /// Counters and timers for parcels received.
        util::high_resolution_timer timer_;
        performance_counters::parcels::data_point receive_data_;
    };

    typedef boost::shared_ptr<parcelport_connection> parcelport_connection_ptr;

    // this makes sure we can store our connections in a set
    inline bool operator<(server::ibverbs::parcelport_connection_ptr const& lhs,
        server::ibverbs::parcelport_connection_ptr const& rhs)
    {
        return lhs.get() < rhs.get();
    }
}}}}

#endif

