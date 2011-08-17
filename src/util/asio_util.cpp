//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <ctime>

#include <boost/system/error_code.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <hpx/util/asio_util.hpp>

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
}}
