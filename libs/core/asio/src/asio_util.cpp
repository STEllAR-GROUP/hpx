//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/system/error_code.hpp
// hpxinspect:nodeprecatedname:boost::system::error_code
// hpxinspect:nodeprecatedinclude:boost/system/system_error.hpp
// hpxinspect:nodeprecatedname:boost::system::system_error

#include <hpx/config.hpp>
#include <hpx/asio/asio_util.hpp>
#include <hpx/util/from_string.hpp>

#include <cstdint>
#include <string>

#include <hpx/config/asio.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/errors.hpp>

#include <asio/io_context.hpp>
#include <asio/ip/address_v4.hpp>
#include <asio/ip/address_v6.hpp>
#include <asio/ip/host_name.hpp>
#include <asio/ip/tcp.hpp>

#include <ctime>
#include <exception>
#include <system_error>

#if defined(HPX_WINDOWS)
// Prevent asio from initializing Winsock, the object must be constructed
// before any Asio's own global objects. With MSVC, this may be accomplished
// by adding the following code to the DLL:

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(push)
#pragma warning(disable : 4073)
#endif
#pragma init_seg(lib)
asio::detail::winsock_init<>::manual manual_winsock_init;
#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(pop)
#endif
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {

    ///////////////////////////////////////////////////////////////////////////
    bool get_endpoint(std::string const& addr, std::uint16_t port,
        asio::ip::tcp::endpoint& ep)
    {
        using namespace asio::ip;
        std::error_code ec;
        address_v4 addr4 = address_v4::from_string(addr.c_str(), ec);
        if (!ec)
        {    // it's an IPV4 address
            ep = tcp::endpoint(address(addr4), port);
            return true;
        }

        address_v6 addr6 = address_v6::from_string(addr.c_str(), ec);
        if (!ec)
        {    // it's an IPV6 address
            ep = tcp::endpoint(address(addr6), port);
            return true;
        }
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////
    std::string get_endpoint_name(asio::ip::tcp::endpoint const& ep)
    {
        return ep.address().to_string();
    }

    ///////////////////////////////////////////////////////////////////////////
    // properly resolve a give host name to the corresponding IP address
    asio::ip::tcp::endpoint resolve_hostname(std::string const& hostname,
        std::uint16_t port, asio::io_context& io_service)
    {
        using asio::ip::tcp;

        // collect errors here
        exception_list errors;

        // try to directly create an endpoint from the address
        try
        {
            tcp::endpoint ep;
            if (util::get_endpoint(hostname, port, ep))
                return ep;
        }
        catch (std::system_error const&)
        {
            errors.add(std::current_exception());
        }

        // it's not an address, try to treat it as a host name
        try
        {
            // resolve the given address
            tcp::resolver resolver(io_service);
            tcp::resolver::query query(hostname, std::to_string(port));

            asio::ip::tcp::resolver::iterator it = resolver.resolve(query);
            HPX_ASSERT(it != asio::ip::tcp::resolver::iterator());
            return *it;
        }
        catch (std::system_error const&)
        {
            errors.add(std::current_exception());
        }

        // report errors
        HPX_THROW_EXCEPTION(network_error, "util::resolve_hostname",
            "{} (while trying to resolve: {}:{})", errors.get_message(),
            hostname, port);
        return tcp::endpoint();
    }

    ///////////////////////////////////////////////////////////////////////////
    // return the public IP address of the local node
    std::string resolve_public_ip_address()
    {
        using asio::ip::tcp;

        // collect errors here
        exception_list errors;

        try
        {
            asio::io_context io_service;
            tcp::resolver resolver(io_service);
            tcp::resolver::query query(asio::ip::host_name(), "");
            tcp::resolver::iterator it = resolver.resolve(query);
            tcp::endpoint endpoint = *it;
            return endpoint.address().to_string();
        }
        catch (std::system_error const&)
        {
            errors.add(std::current_exception());
        }

        // report errors
        HPX_THROW_EXCEPTION(network_error, "util::resolve_public_ip_address",
            "{} (while trying to resolve public ip address)",
            errors.get_message());
        return "";
    }

    ///////////////////////////////////////////////////////////////////////
    // Take an ip v4 or v6 address and "standardize" it for comparison checks
    // note that this code doesn't work as expected if we use the boost
    // inet_pton functions on linux. see issue #2177 for further info
    std::string cleanup_ip_address(std::string const& addr)
    {
        char buf[sizeof(struct in6_addr)];
        int i = 0, domain[2] = {AF_INET, AF_INET6};
        char str[INET6_ADDRSTRLEN];

#if defined(HPX_WINDOWS)
        unsigned long scope_id;
        std::error_code ec;
#endif

        for (i = 0; i < 2; ++i)
        {
#if defined(HPX_WINDOWS)
            int s = asio::detail::socket_ops::inet_pton(
                domain[i], &addr[0], buf, &scope_id, ec);
            if (s > 0 && !ec)
                break;
#else
            int s = inet_pton(domain[i], &addr[0], buf);
            if (s > 0)
                break;
#endif
        }
        if (i == 2)
        {
            HPX_THROW_EXCEPTION(bad_parameter, "cleanup_ip_address",
                "Invalid IP address string");
        }

#if defined(HPX_WINDOWS)
        if (asio::detail::socket_ops::inet_ntop(
                domain[i], buf, str, INET6_ADDRSTRLEN, scope_id, ec) == nullptr)
        {
#else
        if (inet_ntop(domain[i], buf, str, INET6_ADDRSTRLEN) == nullptr)
        {
#endif
            HPX_THROW_EXCEPTION(
                bad_parameter, "cleanup_ip_address", "inet_ntop failure");
        }
        return std::string(str);
    }

    endpoint_iterator_type connect_begin(std::string const& address,
        std::uint16_t port, asio::io_context& io_service)
    {
        using asio::ip::tcp;

        // collect errors here
        exception_list errors;

        std::string port_str(std::to_string(port));

        // try to directly create an endpoint from the address
        try
        {
            tcp::endpoint ep;
            if (util::get_endpoint(address, port, ep))
            {
                return endpoint_iterator_type(
                    tcp::resolver::results_type::create(ep, address, port_str));
            }
        }
        catch (std::system_error const&)
        {
            errors.add(std::current_exception());
        }

        // it's not an address, try to treat it as a host name
        try
        {
            // resolve the given address
            tcp::resolver resolver(io_service);
            tcp::resolver::query query(
                !address.empty() ? address : asio::ip::host_name(), port_str);

            return endpoint_iterator_type(resolver.resolve(query));
        }
        catch (std::system_error const&)
        {
            errors.add(std::current_exception());
        }

        // report errors
        HPX_THROW_EXCEPTION(network_error, "connect_begin",
            "{} (while trying to connect to: {}:{})", errors.get_message(),
            address, port);

        return endpoint_iterator_type();
    }

    endpoint_iterator_type accept_begin(std::string const& address,
        std::uint16_t port, asio::io_context& io_service)
    {
        using asio::ip::tcp;

        // collect errors here
        exception_list errors;

        std::string port_str(std::to_string(port));

        // try to directly create an endpoint from the address
        try
        {
            tcp::endpoint ep;
            if (util::get_endpoint(address, port, ep))
            {
                return endpoint_iterator_type(
                    tcp::resolver::results_type::create(ep, address, port_str));
            }
        }
        catch (std::system_error const&)
        {
            errors.add(std::current_exception());
        }

        // it's not an address, try to treat it as a host name
        try
        {
            // resolve the given address
            tcp::resolver resolver(io_service);
            tcp::resolver::query query(address, port_str);

            return endpoint_iterator_type(resolver.resolve(query));
        }
        catch (std::system_error const&)
        {
            errors.add(std::current_exception());
        }

        // it's not a host name either, create a custom iterator allowing to
        // filter the returned endpoints, for this we use "localhost" as the
        // address to enumerate endpoints
        try
        {
            // resolve the given address
            tcp::resolver resolver(io_service);
            tcp::resolver::query query(asio::ip::host_name(), port_str);

            return endpoint_iterator_type(resolver.resolve(query));
        }
        catch (std::system_error const&)
        {
            errors.add(std::current_exception());
        }

        // report errors
        HPX_THROW_EXCEPTION(network_error, "accept_begin",
            "{} (while trying to resolve: {}:{}))", errors.get_message(),
            address, port);
        return endpoint_iterator_type();
    }
}}    // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util {

    ///////////////////////////////////////////////////////////////////////
    // Addresses are supposed to have the format <hostname>[:port]
    bool split_ip_address(
        std::string const& v, std::string& host, std::uint16_t& port)
    {
        std::string::size_type p = v.find_first_of(':');

        std::string tmp_host;
        std::uint16_t tmp_port = 0;

        try
        {
            if (p != std::string::npos)
            {
                tmp_host = v.substr(0, p);
                tmp_port =
                    hpx::util::from_string<std::uint16_t>(v.substr(p + 1));
            }
            else
            {
                tmp_host = v;
            }

            if (!tmp_host.empty())
            {
                host = tmp_host;
                if (tmp_port)
                    port = tmp_port;
            }
        }
        catch (hpx::util::bad_lexical_cast const& /*e*/)
        {
            // port number is invalid
            return false;
        }
        return true;
    }
}}    // namespace hpx::util
