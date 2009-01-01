//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ASIOUTIL_MAY_16_2008_1212PM)
#define HPX_UTIL_ASIOUTIL_MAY_16_2008_1212PM

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
    
}}

#endif



