//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include "dimension.hpp"

namespace boost { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    // Implementation of the serialization function.
    HPX_COMPONENT_EXPORT void
    serialize(hpx::util::portable_binary_iarchive& ar, sheneos::dimension& dim,
        unsigned int const)
    {
        ar & dim.offset_ & dim.count_ & dim.size_;
    }

    HPX_COMPONENT_EXPORT void
    serialize(hpx::util::portable_binary_oarchive& ar, sheneos::dimension& dim,
        unsigned int const)
    {
        ar & dim.offset_ & dim.count_ & dim.size_;
    }

}}

