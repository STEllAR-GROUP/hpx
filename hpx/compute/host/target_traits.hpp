///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_HOST_TARGET_TRAITS_HPP
#define HPX_COMPUTE_HOST_TARGET_TRAITS_HPP

#include <hpx/config.hpp>
#include <hpx/compute/host/target.hpp>

namespace hpx { namespace compute {
    template <>
    struct target_traits<host::target>
    {
        typedef host::target target_type;

        template <typename T>
        static T & access(host::target const&, T * t, std::size_t pos)
        {
            return *(t + pos);
        }
    };
}}

#endif
