////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_4742616F_CBD8_4D0D_9B92_D150201A10F7)
#define HPX_4742616F_CBD8_4D0D_9B92_D150201A10F7

#include <boost/asio/ip/address_v6.hpp>

#include <boost/functional/hash.hpp>
#include <boost/foreach.hpp>

namespace boost { namespace asio { namespace ip
{
    
std::size_t hash_value(address_v6 const& addr)
{
    typedef address_v6::bytes_type::value_type value_type;
    std::size_t seed = 0;
    BOOST_FOREACH(value_type e, addr.to_bytes()) {
        hash_combine(seed, e);
    }
    return seed;
}

}}}

#endif // HPX_4742616F_CBD8_4D0D_9B92_D150201A10F7

