//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_ASIOUTIL_MAY_16_2008_1212PM)
#define HPX_UTIL_ASIOUTIL_MAY_16_2008_1212PM

#include <boost/asio.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    inline bool
    get_endpoint(std::string const& addr, unsigned short port,
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
    
}}

#endif



