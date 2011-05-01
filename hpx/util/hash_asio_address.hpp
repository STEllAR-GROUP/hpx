////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_4E7EFD09_85CA_48FE_9E22_1BED89BC0299)
#define HPX_4E7EFD09_85CA_48FE_9E22_1BED89BC0299

#include <boost/asio/ip/address.hpp>

#include <hpx/util/hash_asio_address_v4.hpp>
#include <hpx/util/hash_asio_address_v6.hpp>

namespace boost { namespace asio { namespace ip
{

std::size_t hash_value(address const& addr)
{
    std::size_t seed = 0;

    // ipv6
    if (addr.is_v6())
    {
        hash_combine(seed, 6);
        hash_combine(seed, addr.to_v6());
    }
    
    // ipv4
    else
    {
        hash_combine(seed, 4);
        hash_combine(seed, addr.to_v4());
    }

    return seed;
}

}}}

#endif // HPX_4E7EFD09_85CA_48FE_9E22_1BED89BC0299

