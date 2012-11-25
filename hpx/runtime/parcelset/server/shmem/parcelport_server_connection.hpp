//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_SERVER_SHMEM_PARCELPORT_CONNECTION_OV_25_2012_0516PM)
#define HPX_PARCELSET_SERVER_SHMEM_PARCELPORT_CONNECTION_OV_25_2012_0516PM

#include <sstream>
#include <vector>

#include <hpx/runtime/parcelset/server/parcelport_queue.hpp>
#include <hpx/runtime/parcelset/shmem/data_window.hpp>
#include <hpx/runtime/parcelset/shmem/data_buffer.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>

#include <boost/atomic.hpp>
#include <boost/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace server { namespace shmem
{
    /// Represents a single parcelport_connection from a client.
    class parcelport_connection
      : public boost::enable_shared_from_this<parcelport_connection>,
        private boost::noncopyable
    {
    public:
        /// Construct a listening parcelport_connection with the given io_service.
        parcelport_connection(boost::asio::io_service& io_service, 
                parcelport_queue& handler)
          : window_(io_service), in_priority_(0), in_size_(0),
            in_buffer_(), parcels_(handler)
        {}

        ~parcelport_connection()
        {
            // gracefully and portably shutdown the connection
            boost::system::error_code ec;
            window_.close(ec);    // close the socket to give it back to the OS
        }

        /// Get the data window associated with the parcelport_connection.
        parcelset::shmem::data_window& window() { return window_; }

        /// Asynchronously read a data structure from the socket.
        template <typename Handler>
        void async_read(Handler handler)
        {
//             // Store the time of the begin of the read operation
//             receive_data_.time_ = timer_.elapsed_nanoseconds();
//             receive_data_.serialization_time_ = 0;
//             receive_data_.bytes_ = 0;
//             receive_data_.num_parcels_ = 0;
// 
//             // Issue a read operation to read the parcel priority and size.
//             void (parcelport_connection::*f)(boost::system::error_code const&,
//                     boost::tuple<Handler>)
//                 = &parcelport_connection::handle_read_header<Handler>;
// 
//             in_buffer_.reset(new std::vector<char>());
//             in_priority_ = 0;
//             in_size_ = 0;
// 
//             using boost::asio::buffer;
//             std::vector<boost::asio::mutable_buffer> buffers;
//             buffers.push_back(buffer(&in_priority_, sizeof(in_priority_)));
//             buffers.push_back(buffer(&in_size_, sizeof(in_size_)));
// 
// #if defined(__linux) || defined(linux) || defined(__linux__)
//             boost::asio::detail::socket_option::boolean<
//                 IPPROTO_TCP, TCP_QUICKACK> quickack(true);
//             socket_.set_option(quickack);
// #endif
// 
//             boost::asio::async_read(socket_, buffers,
//                 boost::bind(f, shared_from_this(),
//                     boost::asio::placeholders::error,
//                     boost::make_tuple(handler)));
        }

    protected:
        /// Handle a completed read of the message priority and size from the
        /// message header.
        /// The handler is passed using a tuple since boost::bind seems to have
        /// trouble binding a function object created using boost::bind as a
        /// parameter.
        template <typename Handler>
        void handle_read_header(boost::system::error_code const& e,
            boost::tuple<Handler> handler)
        {
//             if (e) {
//                 boost::get<0>(handler)(e);
//             }
//             else {
//                 // Determine the length of the serialized data.
//                 boost::uint64_t inbound_data_size = in_size_;
//                 receive_data_.bytes_ = std::size_t(inbound_data_size);
// 
//                 // Start an asynchronous call to receive the data.
//                 in_buffer_->resize(static_cast<std::size_t>(inbound_data_size));
//                 void (parcelport_connection::*f)(boost::system::error_code const&,
//                         boost::tuple<Handler>)
//                     = &parcelport_connection::handle_read_data<Handler>;
// 
// #if defined(__linux) || defined(linux) || defined(__linux__)
//                 boost::asio::detail::socket_option::boolean<
//                     IPPROTO_TCP, TCP_QUICKACK> quickack(true);
//                 socket_.set_option(quickack);
// #endif
// 
//                 boost::asio::async_read(socket_, 
//                     boost::asio::buffer(*in_buffer_.get()),
//                     boost::bind(f, shared_from_this(),
//                         boost::asio::placeholders::error, handler));
//             }
        }

        /// Handle a completed read of message data.
        template <typename Handler>
        void handle_read_data(boost::system::error_code const& e,
            boost::tuple<Handler> handler)
        {
//             if (e) {
//                 boost::get<0>(handler)(e);
//             }
//             else {
//                 // complete data point and pass it along
//                 receive_data_.time_ = timer_.elapsed_nanoseconds() - 
//                     receive_data_.time_;
// 
//                 // add parcel data to incoming parcel queue
//                 boost::integer::ulittle8_t::value_type priority = in_priority_;
//                 parcels_.add_parcel(in_buffer_,
//                     static_cast<threads::thread_priority>(priority),
//                     receive_data_);
// 
//                 // Inform caller that data has been received ok.
//                 boost::get<0>(handler)(e);
// 
//                 // now send acknowledgement byte
//                 ack_ = true;
//                 boost::asio::async_write(socket_, 
//                     boost::asio::buffer(&ack_, sizeof(ack_)),
//                     boost::bind(&parcelport_connection::handle_write_ack, 
//                         shared_from_this()));
// 
//                 // Issue a read operation to read the parcel priority.
//                 async_read(boost::get<0>(handler));
//             }
        }

        void handle_write_ack() {}

    private:
        /// Data window for the parcelport_connection.
        parcelset::shmem::data_window window_;

        /// buffer for incoming data
        unsigned char in_priority_;
        std::size_t in_size_;
        boost::shared_ptr<parcelset::shmem::data_buffer> in_buffer_;
        bool ack_;

        /// The handler used to process the incoming request.
        parcelport_queue& parcels_;

        /// Counters and timers for parcels received.
        util::high_resolution_timer timer_;
        performance_counters::parcels::data_point receive_data_;
    };

    typedef boost::shared_ptr<parcelport_connection> parcelport_connection_ptr;

    // this makes sure we can store our connections in a set
    inline bool operator<(server::shmem::parcelport_connection_ptr const& lhs, 
        server::shmem::parcelport_connection_ptr const& rhs)
    {
        return lhs.get() < rhs.get();
    }
}}}}

#endif

