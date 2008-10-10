//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_PARCELPORT_CONNECTION_MAY_20_2008_1132PM)
#define HPX_PARCELSET_PARCELPORT_CONNECTION_MAY_20_2008_1132PM

#include <sstream>
#include <vector>

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/integer/endian.hpp>

#include <hpx/runtime/parcelset/server/parcelport_queue.hpp>
#include <hpx/runtime/parcelset/connection_cache.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/container_device.hpp>
#include <hpx/util/util.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset 
{
    /// Represents a single parcelport_connection from a client.
    class parcelport_connection
      : public boost::enable_shared_from_this<parcelport_connection>,
        private boost::noncopyable
    {
    private:
        typedef util::container_device<std::vector<char> > io_device_type;

    public:
        /// Construct a sending parcelport_connection with the given io_service.
        parcelport_connection(boost::asio::io_service& io_service,
                naming::locality const& l, connection_cache& cache)
          : socket_(io_service), there_(l), connection_cache_(cache)
        {
        }

        void set_parcel (parcel const& p)
        {
            // guard against serialization errors
            try {
                // create a special io stream on top of out_buffer_
                out_buffer_.clear();
                boost::iostreams::stream<io_device_type> io(out_buffer_);

                // Serialize the data
                util::portable_binary_oarchive archive(io);
                archive << p;
            }
            catch (std::exception const& e) {
                HPX_OSSTREAM strm;
                strm << "parcelport: parcel serialization failed: " << e.what();
                HPX_THROW_EXCEPTION(no_success, HPX_OSSTREAM_GETSTRING(strm));
                return;
            }
            out_size_ = out_buffer_.size();
        }

        /// Get the socket associated with the parcelport_connection.
        boost::asio::ip::tcp::socket& socket() { return socket_; }

        /// Asynchronously write a data structure to the socket.
        template <typename Handler>
        void async_write(Handler handler)
        {
            // Write the serialized data to the socket. We use "gather-write" 
            // to send both the header and the data in a single write operation.
            std::vector<boost::asio::const_buffer> buffers;
            buffers.push_back(boost::asio::buffer(&out_size_, sizeof(out_size_)));
            buffers.push_back(boost::asio::buffer(out_buffer_));

            // this additional wrapping of the handler into a bind object is 
            // needed to keep  this parcelport_connection object alive for the whole
            // write operation
            void (parcelport_connection::*f)(boost::system::error_code const&, std::size_t,
                    boost::tuple<Handler>)
                = &parcelport_connection::handle_write<Handler>;

            boost::asio::async_write(socket_, buffers,
                boost::bind(f, shared_from_this(), 
                    boost::asio::placeholders::error, _2, 
                    boost::make_tuple(handler)));
        }

    protected:
        /// handle completed write operation
        template <typename Handler>
        void handle_write(boost::system::error_code const& e, std::size_t bytes,
            boost::tuple<Handler> handler)
        {
            if (e) {
                LPT_(error) << "parcelhandler: put parcel failed: " 
                            << e.message();
            }
            else {
                LPT_(info) << "parcelhandler: put parcel succeeded";
            }

            // just call initial handler
            boost::get<0>(handler)(e, bytes);

            // now we can give this connection back to the cache
            out_buffer_.clear();
            out_size_ = 0;
            connection_cache_.add(there_, shared_from_this());
        }

    private:
        /// Socket for the parcelport_connection.
        boost::asio::ip::tcp::socket socket_;

        /// buffer for outgoing data
        boost::integer::ulittle64_t out_size_;
        std::vector<char> out_buffer_;

        /// the other (receiving) end of this connection
        naming::locality there_;

        /// The connection cache for sending connections
        connection_cache& connection_cache_;
    };

    typedef boost::shared_ptr<parcelport_connection> parcelport_connection_ptr;

///////////////////////////////////////////////////////////////////////////////
}}

#endif
