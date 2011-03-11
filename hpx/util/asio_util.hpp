//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ASIOUTIL_MAY_16_2008_1212PM)
#define HPX_UTIL_ASIOUTIL_MAY_16_2008_1212PM

#include <ctime>

#include <hpx/config.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/asio/ip/tcp.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    bool get_endpoint(std::string const& addr, boost::uint16_t port,
        boost::asio::ip::tcp::endpoint& ep);

    ///////////////////////////////////////////////////////////////////////////
    typedef 
        std::pair<
            boost::asio::ip::tcp::resolver::iterator, 
            boost::asio::ip::tcp::resolver::iterator
        >
    tcp_iterator_range_type;

    tcp_iterator_range_type accept();
    tcp_iterator_range_type connect();

    inline boost::fusion::vector2<boost::uint16_t, boost::uint16_t>
    get_random_ports()
    {
        boost::mt19937 rng((boost::uint32_t)std::time(NULL));
        boost::uniform_int<boost::uint16_t>
            port_range(HPX_RANDOM_PORT_MIN, HPX_RANDOM_PORT_MAX-1);

        boost::uint16_t p = port_range(rng);
        return boost::fusion::vector2<boost::uint16_t, boost::uint16_t>
            (p, p+1); 
    }
}}

#endif

