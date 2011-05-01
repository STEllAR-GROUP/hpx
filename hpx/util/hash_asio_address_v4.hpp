////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_6775A2B5_13D9_4C46_9617_CE96B16CFCF2)
#define HPX_6775A2B5_13D9_4C46_9617_CE96B16CFCF2

#include <boost/asio/ip/address_v4.hpp>

#include <boost/functional/hash.hpp>
#include <boost/foreach.hpp>

namespace boost { namespace asio { namespace ip
{
    
std::size_t hash_value(address_v4 const& addr)
{
    typedef address_v4::bytes_type::value_type value_type;
    std::size_t seed = 0;
    BOOST_FOREACH(value_type e, addr.to_bytes()) {
        hash_combine(seed, e);
    }
    return seed;
}

}}}

#endif // HPX_6775A2B5_13D9_4C46_9617_CE96B16CFCF2

