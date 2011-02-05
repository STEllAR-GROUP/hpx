//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
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

}}
