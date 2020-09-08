//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstdint>
#include <string>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/config/asio.hpp>

#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>
/* The boost asio support includes termios.h.
 * The termios.h file on ppc64le defines these macros, which
 * are also used by blaze, blaze_tensor as Template names.
 * Make sure we undefine them before continuing. */
#undef VT1
#undef VT2

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    HPX_CORE_EXPORT bool get_endpoint(std::string const& addr,
        std::uint16_t port, boost::asio::ip::tcp::endpoint& ep);

    HPX_CORE_EXPORT std::string get_endpoint_name(
        boost::asio::ip::tcp::endpoint const& ep);

    ///////////////////////////////////////////////////////////////////////////
    // properly resolve a give host name to the corresponding IP address
    HPX_CORE_EXPORT boost::asio::ip::tcp::endpoint resolve_hostname(
        std::string const& hostname, std::uint16_t port,
        boost::asio::io_service& io_service);

    ///////////////////////////////////////////////////////////////////////////
    // return the public IP address of the local node
    HPX_CORE_EXPORT std::string resolve_public_ip_address();

    ///////////////////////////////////////////////////////////////////////
    // Take an ip v4 or v6 address and "standardize" it for comparison checks
    HPX_CORE_EXPORT std::string cleanup_ip_address(const std::string& addr);

    typedef boost::asio::ip::tcp::resolver::iterator endpoint_iterator_type;

    endpoint_iterator_type HPX_CORE_EXPORT connect_begin(
        std::string const& address, std::uint16_t port,
        boost::asio::io_service& io_service);

    /// \brief Returns an iterator which when dereferenced will give an
    ///        endpoint suitable for a call to connect() related to this
    ///        locality
    template <typename Locality>
    endpoint_iterator_type connect_begin(
        Locality const& loc, boost::asio::io_service& io_service)
    {
        return connect_begin(loc.address(), loc.port(), io_service);
    }

    inline endpoint_iterator_type HPX_CORE_EXPORT connect_end()
    {
        return endpoint_iterator_type();
    }

    endpoint_iterator_type HPX_CORE_EXPORT accept_begin(
        std::string const& address, std::uint16_t port,
        boost::asio::io_service& io_service);

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Returns an iterator which when dereferenced will give an
    ///        endpoint suitable for a call to accept() related to this
    ///        locality
    template <typename Locality>
    endpoint_iterator_type accept_begin(
        Locality const& loc, boost::asio::io_service& io_service)
    {
        return accept_begin(loc.address(), loc.port(), io_service);
    }

    inline endpoint_iterator_type accept_end()    //-V524
    {
        return endpoint_iterator_type();
    }
}}    // namespace hpx::util

#endif

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////
    // Addresses are supposed to have the format <hostname>[:port]
    HPX_CORE_EXPORT bool split_ip_address(
        std::string const& v, std::string& host, std::uint16_t& port);
}}    // namespace hpx::util
