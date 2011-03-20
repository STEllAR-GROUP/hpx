////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_840335F1_3241_4F72_86EB_9331AA5AA7DE)
#define HPX_840335F1_3241_4F72_86EB_9331AA5AA7DE

#include <boost/asio/ip/address_v6.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/split_free.hpp>

namespace boost { namespace serialization
{

template <typename Archive>
void save(Archive& ar, asio::ip::address_v6 const& v, unsigned int)
{
    asio::ip::address_v6::bytes_type bytes = v.to_bytes();
    ar & bytes;
}

template <typename Archive>
void load(Archive& ar, asio::ip::address_v6& v, unsigned int)
{
    asio::ip::address_v6::bytes_type bytes;
    ar & bytes;
    v = asio::ip::address_v6(bytes);
}

}}

BOOST_SERIALIZATION_SPLIT_FREE(boost::asio::ip::address_v6);

#endif // HPX_840335F1_3241_4F72_86EB_9331AA5AA7DE

