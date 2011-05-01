////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_2F246290_3851_4218_95E5_66F823D08141)
#define HPX_2F246290_3851_4218_95E5_66F823D08141

#include <boost/asio/ip/basic_endpoint.hpp>

#include <hpx/util/hash_asio_address.hpp>

namespace boost { namespace asio { namespace ip
{

template <typename InternetProtocol>
std::size_t hash_value(basic_endpoint<InternetProtocol> const& ep)
{
    std::size_t seed = 0;
    hash_combine(seed, ep.port());
    hash_combine(seed, ep.address());
    return seed;
}

}}}

#endif // HPX_2F246290_3851_4218_95E5_66F823D08141

