////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_B01E19EB_56A8_4918_806F_DB7088BC081B)
#define HPX_B01E19EB_56A8_4918_806F_DB7088BC081B

#include <boost/asio/ip/address.hpp>

#include <hpx/util/serialize_asio_address_v4.hpp>
#include <hpx/util/serialize_asio_address_v6.hpp>

namespace boost { namespace serialization
{

template <typename Archive>
void save(Archive& ar, asio::ip::address const& v, unsigned int)
{
    // ipv6
    if (v.is_v6())
        ar & 1 & v.to_v6();
    // ipv4
    ar & 0 & v.to_v4(); 
}

template <typename Archive>
void load(Archive& ar, asio::ip::address& v, unsigned int)
{
    bool proto;

    ar & proto;

    // ipv6
    if (proto == 1)
    {
        asio::ip::address_v6 addr;
        ar & addr;
        v = addr;
    }
    // ipv4
    asio::ip::address_v4 addr;     
    ar & addr;
    v = addr; 
}

}}

BOOST_SERIALIZATION_SPLIT_FREE(boost::asio::ip::address);

#endif // HPX_B01E19EB_56A8_4918_806F_DB7088BC081B

