//  Copyright (c) 2007-2010 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_NAMING_RESOLVERCLIENT_CONNECTION_MAY_27_2008_0317PM)
#define HPX_NAMING_RESOLVERCLIENT_CONNECTION_MAY_27_2008_0317PM

#include <sstream>
#include <vector>

#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/container_device.hpp>
#include <hpx/runtime/naming/server/reply.hpp>

#include <boost/asio/io_service.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/bind.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/integer/endian.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace naming 
{
    /// Represents a single resolver_client_connection from a client.
    struct resolver_client_connection
      : public boost::enable_shared_from_this<resolver_client_connection>,
        private boost::noncopyable
    {
    public:
        /// Construct a sending resolver_client_connection (for the \a
        /// naming#server#command_resolve command)
        resolver_client_connection(boost::asio::io_service& io_service)
          : socket_(io_service)
        {}

        /// Get the socket associated with the resolver_client_connection.
        boost::asio::ip::tcp::socket& socket() { return socket_; }

    private:
        /// Socket for the resolver_client_connection.
        boost::asio::ip::tcp::socket socket_;
    };

///////////////////////////////////////////////////////////////////////////////
}}

#endif
