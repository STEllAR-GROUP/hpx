////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_B01E19EB_56A8_4918_806F_DB7088BC081B)
#define HPX_B01E19EB_56A8_4918_806F_DB7088BC081B

#include <boost/asio/ip/address.hpp>

// add serialization support for std::array
#if defined(BOOST_ASIO_HAS_STD_ARRAY)
#include <boost/serialization/array.hpp>

namespace boost { namespace serialization
{
    template <class Archive, class T, std::size_t N>
    void serialize(Archive& ar, std::array<T, N>& a, const unsigned int /* version */)
    {
        ar & boost::serialization::make_array(a.data(), N);
    }
}}
#endif

#include <hpx/util/serialize_asio_address_v4.hpp>
#include <hpx/util/serialize_asio_address_v6.hpp>

namespace boost { namespace serialization
{

template <typename Archive>
void save(Archive& ar, asio::ip::address const& v, unsigned int)
{
    // ipv6
    if (v.is_v6())
    {
        bool i = 1; // serialization is silly, won't take const references
        asio::ip::address_v6 a = v.to_v6(); // see: above
        ar & i & a;
    }
    
    // ipv4
    else
    {
        bool i = 0;
        asio::ip::address_v4 a = v.to_v4();
        ar & i & a;
    } 
}

template <typename Archive>
void load(Archive& ar, asio::ip::address& v, unsigned int)
{
    bool proto;
    ar & proto;

    // ipv6
    if (proto)
    {
        asio::ip::address_v6 addr;
        ar & addr;
        v = addr;
    }

    // ipv4
    else
    {
        asio::ip::address_v4 addr;     
        ar & addr;
        v = addr;
    }
}

}}

BOOST_SERIALIZATION_SPLIT_FREE(boost::asio::ip::address);

#endif // HPX_B01E19EB_56A8_4918_806F_DB7088BC081B

