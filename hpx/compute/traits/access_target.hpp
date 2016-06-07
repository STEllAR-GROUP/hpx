///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_TARGET_TRAITS_HPP
#define HPX_COMPUTE_TARGET_TRAITS_HPP

#include <hpx/config.hpp>

namespace hpx { namespace compute { namespace traits
{
    template <typename Target, typename Enable = void>
    struct access_target;
}}}

#endif
