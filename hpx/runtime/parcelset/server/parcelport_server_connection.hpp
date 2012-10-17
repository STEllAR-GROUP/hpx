//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2010-2012 Bryce Lelbach
//  Copyright (c) 2011 Katelyn Kufahl
//
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_SERVER_PARCELPORT_CONNECTION_MAR_26_2008_1221PM)
#define HPX_PARCELSET_SERVER_PARCELPORT_CONNECTION_MAR_26_2008_1221PM

#include <sstream>
#include <vector>

#include <hpx/runtime/parcelset/server/parcelport_queue.hpp>
#include <hpx/util/zero_copy_input.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>

#include <boost/asio/buffer.hpp>
#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/atomic.hpp>
#include <boost/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/integer/endian.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/tuple/tuple.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset { namespace server
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
          : socket_(io_service), parcel_queue_(handler)
        {}

        ~parcelport_connection()
        {
            // gracefully and portably shutdown the socket
            boost::system::error_code ec;
            socket_.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
            socket_.close(ec);    // close the socket to give it back to the OS
        }

        /// Get the socket associated with the parcelport_connection.
        boost::asio::ip::tcp::socket& socket() { return socket_; }

        void enable_tcp_quickack() 
        {
#if defined(__linux) || defined(linux) || defined(__linux__)
            boost::asio::detail::socket_option::boolean<
                IPPROTO_TCP, TCP_QUICKACK> quickack(true);
            socket_.set_option(quickack);
#endif
        }

        /// Asynchronously read a data structure from the socket.
        template <typename Handler>
        void async_read(Handler handler)
        {
            // Store the time of the begin of the read operation
            receive_data_.time_ = timer_.elapsed_nanoseconds();
            receive_data_.serialization_time_ = 0;
            receive_data_.bytes_ = 0;
            receive_data_.num_parcels_ = 0;

            // Re-initialize the input data structures for the incomming message
            input_data_.prepare();

            // Issue a read operation to read the parcel priority and size.
            void (parcelport_connection::*f)(boost::system::error_code const&,
                    boost::tuple<Handler>)
                = &parcelport_connection::handle_read_header<Handler>;

            enable_tcp_quickack();
            boost::asio::async_read(socket_, input_data_.get_buffers(),
                boost::bind(f, shared_from_this(),
                    boost::asio::placeholders::error,
                    boost::make_tuple(handler)));
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
            if (e) {
                boost::get<0>(handler)(e);
            }
            else {
                // kick off reading of the chunk sizes
                input_data_.prepare_chunks();

                void (parcelport_connection::*f)(boost::system::error_code const&,
                        boost::tuple<Handler>)
                    = &parcelport_connection::handle_read_chunks<Handler>;

                enable_tcp_quickack();
                boost::asio::async_read(socket_, input_data_.get_buffers(),
                    boost::bind(f, shared_from_this(),
                        boost::asio::placeholders::error, handler));
            }
        }

        // we've got the number of chunks, now read the real data
        template <typename Handler>
        void handle_read_chunks(boost::system::error_code const& e,
            boost::tuple<Handler> handler)
        {
            if (e) {
                boost::get<0>(handler)(e);
            }
            else {
                // We've got the chunk sizes, now prepare the buffers and 
                // read the actual data, this fills the array of buffers.
                input_data_.load_pass1();

                // Start an asynchronous call to receive the data.
                void (parcelport_connection::*f)(boost::system::error_code const&,
                        boost::tuple<Handler>)
                    = &parcelport_connection::handle_read_data<Handler>;

                enable_tcp_quickack();
                boost::asio::async_read(socket_, input_data_.get_buffers(),
                    boost::bind(f, shared_from_this(),
                        boost::asio::placeholders::error, handler));
            }
        }

        /// Handle a completed read of message data.
        template <typename Handler>
        void handle_read_data(boost::system::error_code const& e,
            boost::tuple<Handler> handler)
        {
            if (e) {
                boost::get<0>(handler)(e);
            }
            else {
                // complete data point and pass it along, store time 
                // required to receive data
                receive_data_.time_ = timer_.elapsed_nanoseconds() - 
                    receive_data_.time_;

                try {
                    try {
                        // mark start of serialization
                        util::high_resolution_timer timer;

                        // We've got the data, now deserialize the parcels
                        boost::shared_ptr<std::vector<parcel> > parcels(
                            new std::vector<parcel>());
                        input_data_.load_pass2(*parcels);

                        // add parcel data to incoming parcel queue
                        boost::integer::ulittle8_t::value_type priority = 
                            input_data_.priority();

                        // store the time required for serialization and parcel count
                        receive_data_.serialization_time_ = timer.elapsed_nanoseconds();
                        receive_data_.num_parcels_ = parcels->size();

                        parcel_queue_.add_parcel(parcels,
                            static_cast<threads::thread_priority>(priority),
                            receive_data_);
                    }
                    catch (boost::system::system_error const& e) {
                        LPT_(error)
                            << "decode_parcel: caught boost::system::error: "
                            << e.what();
                        parcel_queue_.add_exception(boost::current_exception());
                    }
                    catch (boost::exception const&) {
                        LPT_(error)
                            << "decode_parcel: caught boost::exception.";
                        parcel_queue_.add_exception(boost::current_exception());
                    }
                    catch (std::exception const& e) {
                        // We have to repackage all exceptions thrown by the
                        // serialization library as otherwise we will loose the
                        // e.what() description of the problem, due to slicing.
                        boost::throw_exception(boost::enable_error_info(
                            hpx::exception(serialization_error, e.what())));
                    }
                }
                catch (...) {
                    // just inform the parcel queue of the error
                    parcel_queue_.add_exception(boost::current_exception());
                }

                // Inform caller that data has been received ok.
                boost::get<0>(handler)(e);

                // now send acknowledgement byte
                ack_ = true;
                boost::asio::async_write(socket_, 
                    boost::asio::buffer(&ack_, sizeof(ack_)),
                    boost::bind(&parcelport_connection::handle_write_ack, 
                        shared_from_this()));

                // Issue a read operation to read the parcel priority.
                async_read(boost::get<0>(handler));
            }
        }

        void handle_write_ack() {}

    private:
        /// Socket for the parcelport_connection.
        boost::asio::ip::tcp::socket socket_;

        /// buffer for incoming data
        util::zero_copy_input input_data_;

//         boost::integer::ulittle8_t in_priority_;
//         boost::integer::ulittle64_t in_size_;
//         boost::shared_ptr<std::vector<char> > in_buffer_;
        bool ack_;

        /// The handler used to process the incoming request.
        parcelport_queue& parcel_queue_;

        /// Counters and timers for parcels received.
        util::high_resolution_timer timer_;
        performance_counters::parcels::data_point receive_data_;
    };

    typedef boost::shared_ptr<parcelport_connection> parcelport_connection_ptr;

    // this makes sure we can store our connections in a set
    inline bool operator<(server::parcelport_connection_ptr const& lhs, 
        server::parcelport_connection_ptr const& rhs)
    {
        return lhs.get() < rhs.get();
    }
}}}

#endif

