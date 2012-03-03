//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/util/asio_util.hpp>

#include <ctime>

#include <boost/system/error_code.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/lexical_cast.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    bool get_endpoint(std::string const& addr, boost::uint16_t port,
        boost::asio::ip::tcp::endpoint& ep)
    {
        using namespace boost::asio::ip;
        boost::system::error_code ec;
        address_v4 addr4 = address_v4::from_string(addr.c_str(), ec);
        if (!ec) {  // it's an IPV4 address
            ep = tcp::endpoint(address(addr4), port);
            return true;
        }

        address_v6 addr6 = address_v6::from_string(addr.c_str(), ec);
        if (!ec) {  // it's an IPV6 address
            ep = tcp::endpoint(address(addr6), port);
            return true;
        }
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////
    std::string get_endpoint_name(boost::asio::ip::tcp::endpoint const& ep)
    {
        return ep.address().to_string();
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::fusion::vector2<boost::uint16_t, boost::uint16_t>
    get_random_ports()
    {
        boost::mt19937 rng((boost::uint32_t)std::time(NULL));
        boost::uniform_int<boost::uint16_t>
            port_range(HPX_RANDOM_PORT_MIN, HPX_RANDOM_PORT_MAX-1);

        boost::uint16_t p = port_range(rng);
        return boost::fusion::vector2<boost::uint16_t, boost::uint16_t>(p, p+1);
    }

    boost::uint16_t
    get_random_port()
    {
        boost::mt19937 rng((boost::uint32_t)std::time(NULL));
        boost::uniform_int<boost::uint16_t>
            port_range(HPX_RANDOM_PORT_MIN, HPX_RANDOM_PORT_MAX-1);

        return port_range(rng);
    }

    ///////////////////////////////////////////////////////////////////////////
    // properly resolve a give host name to the corresponding IP address
    boost::asio::ip::tcp::endpoint
    resolve_hostname(std::string const& hostname, boost::uint16_t port,
        boost::asio::io_service& io_service)
    {
        using boost::asio::ip::tcp;

        // collect errors here
        exception_list errors;

        // try to directly create an endpoint from the address
        try {
            tcp::endpoint ep;
            if (util::get_endpoint(hostname, port, ep))
                return ep;
        }
        catch (boost::system::system_error const& e) {
            errors.add(e);
        }

        // it's not an address, try to treat it as a host name
        try {
            // resolve the given address
            tcp::resolver resolver(io_service);
            tcp::resolver::query query(hostname,
                boost::lexical_cast<std::string>(port));

            boost::asio::ip::tcp::resolver::iterator it =
                resolver.resolve(query);
            BOOST_ASSERT(it != boost::asio::ip::tcp::resolver::iterator());
            return *it;
        }
        catch (boost::system::system_error const& e) {
            errors.add(e);
        }

        // report errors
        hpx::util::osstream strm;
        strm << errors.get_message() << " (while trying to resolve: "
             << hostname << ":" << port << ")";
        HPX_THROW_EXCEPTION(network_error, "util::resolve_hostname",
            hpx::util::osstream_get_string(strm));
        return tcp::endpoint();
    }

    ///////////////////////////////////////////////////////////////////////
    // Addresses are supposed to have the format <hostname>[:port]
    void split_ip_address(std::string const& v, std::string& host,
        boost::uint16_t& port)
    {
        std::string::size_type p = v.find_first_of(":");

        std::string tmp_host;
        boost::uint16_t tmp_port = HPX_INITIAL_IP_PORT;

        try {
            if (p != std::string::npos) {
                tmp_host = v.substr(0, p);
                tmp_port = boost::lexical_cast<boost::uint16_t>(v.substr(p+1));
            }
            else {
                tmp_host = v;
            }

            if (!tmp_host.empty()) {
                host = tmp_host;
                if (tmp_port)
                    port = tmp_port;
            }
        }
        catch (boost::bad_lexical_cast const& /*e*/) {
            std::cerr << "hpx::init: illegal port number given: "
                      << v.substr(p+1)
                      << ", using default value instead: "
                      << port << std::endl;
        }
    }
}}
