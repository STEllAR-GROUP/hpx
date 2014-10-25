//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_ACCESS_HPP
#define HPX_SERIALIZATION_ACCESS_HPP

#include <hpx/config.hpp>

namespace hpx { namespace serialization {
    struct access
    {
        template <typename Archive, typename T>
        static void serialize(Archive & ar, T & t, unsigned)
        {
            t.serialize(ar, 0);
        }
    };
}}

#endif
