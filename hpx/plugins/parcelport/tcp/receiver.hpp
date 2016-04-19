//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2011 Katelyn Kufahl
//  Copyright (c) 2011 Bryce Lelbach
//
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_TCP_RECEIVER_HPP
#define HPX_PARCELSET_POLICIES_TCP_RECEIVER_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_TCP)

#include <hpx/config/asio.hpp>
#include <hpx/runtime/parcelset/parcelport_connection.hpp>
#include <hpx/runtime/parcelset/decode_parcels.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/protect.hpp>

#include <boost/asio/buffer.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/atomic.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

#include <mutex>
#include <sstream>
#include <vector>

namespace hpx { namespace parcelset { namespace policies { namespace tcp
{
    class connection_handler;

    class receiver
      : public parcelport_connection<receiver, std::vector<char>, std::vector<char> >
    {
        typedef hpx::lcos::local::spinlock mutex_type;
    public:
        receiver(boost::asio::io_service& io_service, boost::uint64_t max_inbound_size,
            connection_handler& parcelport)
          : socket_(io_service)
          , max_inbound_size_(max_inbound_size)
          , ack_(0)
          , parcelport_(parcelport)
        {}

        ~receiver()
        {
            shutdown();
        }

        /// Get the socket associated with the parcelport_connection.
        boost::asio::ip::tcp::socket& socket() { return socket_; }

        /// Asynchronously read a data structure from the socket.
        template <typename Handler>
        void async_read(Handler handler)
        {
            HPX_ASSERT(buffer_.data_.empty());
            // Store the time of the begin of the read operation
            performance_counters::parcels::data_point& data = buffer_.data_point_;
            data.time_ = timer_.elapsed_nanoseconds();
            data.serialization_time_ = 0;
            data.bytes_ = 0;
            data.num_parcels_ = 0;

            // Issue a read operation to read the message size.
            using boost::asio::buffer;
            std::vector<boost::asio::mutable_buffer> buffers;
            buffers.push_back(buffer(&buffer_.size_,
                sizeof(buffer_.size_)));
            buffers.push_back(buffer(&buffer_.data_size_,
                sizeof(buffer_.data_size_)));

            buffers.push_back(buffer(&buffer_.num_chunks_,
                sizeof(buffer_.num_chunks_)));

            {
                std::unique_lock<mutex_type> lk(mtx_);
                if(!socket_.is_open())
                {
                    lk.unlock();
                    // report this problem back to the handler
                    handler(boost::asio::error::make_error_code(
                        boost::asio::error::not_connected));
                    return;
                }
#if defined(__linux) || defined(linux) || defined(__linux__)
                boost::asio::detail::socket_option::boolean<
                    IPPROTO_TCP, TCP_QUICKACK> quickack(true);
                socket_.set_option(quickack);
#endif

                void (receiver::*f)(boost::system::error_code const&,
                        std::size_t, Handler)
                    = &receiver::handle_read_header<Handler>;

                boost::asio::async_read(socket_, buffers,
                    util::bind(f, shared_from_this(),
                        boost::asio::placeholders::error,
                        boost::asio::placeholders::bytes_transferred,
                        util::protect(handler)));
            }
        }

        void shutdown()
        {
            std::lock_guard<mutex_type> lk(mtx_);
            // gracefully and portably shutdown the socket
            boost::system::error_code ec;
            if (socket_.is_open()) {
                socket_.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
                socket_.close(ec);    // close the socket to give it back to the OS
            }
        }

    private:
        /// Handle a completed read of the message size from the
        /// message header.
        template <typename Handler>
        void handle_read_header(boost::system::error_code const& e,
            std::size_t bytes_transferred, Handler handler)
        {
            if (e) {
                handler(e);

                // Issue a read operation to read the next parcel.
//                 async_read(handler);
            }
            else {
                // Determine the length of the serialized data.
                boost::uint64_t inbound_size = buffer_.size_;

                if (inbound_size > max_inbound_size_)
                {
                    // report this problem back to the handler
                    handler(boost::asio::error::make_error_code(
                        boost::asio::error::operation_not_supported));
                    return;
                }

                buffer_.data_point_.bytes_ = static_cast<std::size_t>(inbound_size);

                // receive buffers
                std::vector<boost::asio::mutable_buffer> buffers;

                // determine the size of the chunk buffer
                std::size_t num_zero_copy_chunks =
                    static_cast<std::size_t>(
                        static_cast<boost::uint32_t>(buffer_.num_chunks_.first));
                std::size_t num_non_zero_copy_chunks =
                    static_cast<std::size_t>(
                        static_cast<boost::uint32_t>(buffer_.num_chunks_.second));

                void (receiver::*f)(boost::system::error_code const&,
                        Handler) = 0;

                if (num_zero_copy_chunks != 0) {
                    typedef parcel_buffer_type::transmission_chunk_type
                        transmission_chunk_type;

                    std::vector<transmission_chunk_type>& chunks =
                        buffer_.transmission_chunks_;

                    chunks.resize(static_cast<std::size_t>(
                        num_zero_copy_chunks + num_non_zero_copy_chunks));

                    buffers.push_back(
                        boost::asio::buffer(chunks.data(), chunks.size() *
                            sizeof(transmission_chunk_type)));

                    // add main buffer holding data which was serialized normally
                    buffer_.data_.resize(static_cast<std::size_t>(inbound_size));
                    buffers.push_back(boost::asio::buffer(buffer_.data_));

                    // Start an asynchronous call to receive the data.
                    f = &receiver::handle_read_chunk_data<Handler>;
                }
                else {
                    // add main buffer holding data which was serialized normally
                    buffer_.data_.resize(static_cast<std::size_t>(inbound_size));
                    buffers.push_back(boost::asio::buffer(buffer_.data_));

                    // Start an asynchronous call to receive the data.
                    f = &receiver::handle_read_data<Handler>;
                }

                {
                    std::unique_lock<mutex_type> lk(mtx_);
                    if(!socket_.is_open())
                    {
                        lk.unlock();
                        // report this problem back to the handler
                        handler(boost::asio::error::make_error_code(
                            boost::asio::error::not_connected));
                        return;
                    }
#if defined(__linux) || defined(linux) || defined(__linux__)
                    boost::asio::detail::socket_option::boolean<
                        IPPROTO_TCP, TCP_QUICKACK> quickack(true);
                    socket_.set_option(quickack);
#endif
                    boost::asio::async_read(socket_, buffers,
                        util::bind(f, shared_from_this(),
                            boost::asio::placeholders::error,
                            util::protect(handler)));
                }
            }
        }

        /// Handle a completed read of message data.
        template <typename Handler>
        void handle_read_chunk_data(boost::system::error_code const& e,
            Handler handler)
        {
            if (e) {
                handler(e);

                // Issue a read operation to read the next parcel.
//                 async_read(handler);
            }
            else {
                // receive buffers
                std::vector<boost::asio::mutable_buffer> buffers;

                // add appropriately sized chunk buffers for the zero-copy data
                std::size_t num_zero_copy_chunks =
                    static_cast<std::size_t>(
                        static_cast<boost::uint32_t>(buffer_.num_chunks_.first));

                buffer_.chunks_.resize(num_zero_copy_chunks);
                for (std::size_t i = 0; i != num_zero_copy_chunks; ++i)
                {
                    std::size_t chunk_size = buffer_.transmission_chunks_[i].second;
                    buffer_.chunks_[i].resize(chunk_size);
                    buffers.push_back(
                        boost::asio::buffer(buffer_.chunks_[i].data(), chunk_size));
                }

                // Start an asynchronous call to receive the data.
                void (receiver::*f)(boost::system::error_code const&,
                        Handler)
                    = &receiver::handle_read_data<Handler>;

                {
                    std::unique_lock<mutex_type> lk(mtx_);
                    if(!socket_.is_open())
                    {
                        lk.unlock();
                        // report this problem back to the handler
                        handler(boost::asio::error::make_error_code(
                            boost::asio::error::not_connected));
                        return;
                    }
#if defined(__linux) || defined(linux) || defined(__linux__)
                    boost::asio::detail::socket_option::boolean<
                        IPPROTO_TCP, TCP_QUICKACK> quickack(true);
                    socket_.set_option(quickack);
#endif
                    boost::asio::async_read(socket_, buffers,
                        util::bind(f, shared_from_this(),
                            boost::asio::placeholders::error,
                            util::protect(handler)));
                }
            }
        }

        /// Handle a completed read of message data.
        template <typename Handler>
        void handle_read_data(boost::system::error_code const& e,
            Handler handler)
        {
            if (e) {
                handler(e);

                // Issue a read operation to read the next parcel.
//                 async_read(handler);
            }
            else {
                // complete data point and pass it along
                buffer_.data_point_.time_ = timer_.elapsed_nanoseconds() -
                    buffer_.data_point_.time_;

                // now send acknowledgment byte
                void (receiver::*f)(boost::system::error_code const&,
                        Handler)
                    = &receiver::handle_write_ack<Handler>;

                // decode the received parcels.
                decode_parcels(parcelport_, std::move(buffer_), -1);
                buffer_ = parcel_buffer_type();

                ack_ = true;
                {
                    std::unique_lock<mutex_type> lk(mtx_);
                    if(!socket_.is_open())
                    {
                        lk.unlock();
                        // report this problem back to the handler
                        handler(boost::asio::error::make_error_code(
                            boost::asio::error::not_connected));
                        return;
                    }
                    boost::asio::async_write(socket_,
                        boost::asio::buffer(&ack_, sizeof(ack_)),
                        util::bind(f, shared_from_this(),
                            boost::asio::placeholders::error,
                            util::protect(handler)));
                }
            }
        }

        template <typename Handler>
        void handle_write_ack(boost::system::error_code const& e,
            Handler handler)
        {
            // Inform caller that data has been received ok.
            handler(e);

            // Issue a read operation to read the next parcel.
            if (!e)
            {
                async_read(handler);
            }
        }


        /// Socket for the parcelport_connection.
        boost::asio::ip::tcp::socket socket_;

        boost::uint64_t max_inbound_size_;

        bool ack_;

        /// The handler used to process the incoming request.
        connection_handler& parcelport_;

        /// Counters and timers for parcels received.
        util::high_resolution_timer timer_;

        mutex_type mtx_;
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
