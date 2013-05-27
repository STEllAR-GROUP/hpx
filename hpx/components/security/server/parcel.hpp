//  Copyright (c) 2013 Jeroen Habraken
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPONENTS_SECURITY_SERVER_PARCEL_HPP
#define HPX_COMPONENTS_SECURITY_SERVER_PARCEL_HPP

#include <boost/cstdint.hpp>
#include <hpx/hpx_fwd.hpp>

#include "hash.hpp"

namespace hpx { namespace components { namespace security { namespace server
{
    class parcel
    {
    public:
        parcel()
        {
        }

        parcel(boost::uint64_t nonce, hash & hash)
          : nonce_(nonce), hash_(hash.final())
        {
        }

    private:
        boost::uint64_t nonce_;
        traits::hash<>::final_type hash_;
    };
}}}}

#endif
