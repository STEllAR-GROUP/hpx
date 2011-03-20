////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_AB50970C_06F8_4B5E_9FEC_CFACA825175F)
#define HPX_AB50970C_06F8_4B5E_9FEC_CFACA825175F

#include <boost/asio/ip/address_v4.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/split_free.hpp>

namespace boost { namespace serialization
{

template <typename Archive>
void save(Archive& ar, asio::ip::address_v4 const& v, unsigned int)
{
    asio::ip::address_v4::bytes_type bytes = v.to_bytes();
    ar & bytes;
}

template <typename Archive>
void load(Archive& ar, asio::ip::address_v4& v, unsigned int)
{
    asio::ip::address_v4::bytes_type bytes;
    ar & bytes;
    v = asio::ip::address_v4(bytes);
}

}}

BOOST_SERIALIZATION_SPLIT_FREE(boost::asio::ip::address_v4);

#endif // HPX_AB50970C_06F8_4B5E_9FEC_CFACA825175F

