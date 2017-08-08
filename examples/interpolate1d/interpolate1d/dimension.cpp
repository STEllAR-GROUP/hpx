//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/include/serialization.hpp>

#include "dimension.hpp"

namespace hpx { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////////
    // implement the serialization functions
    template <typename Archive>
    void serialize(Archive& ar, interpolate1d::dimension& dim, unsigned int const)
    {
        ar & dim.offset_ & dim.count_ & dim.size_;
    }

    ///////////////////////////////////////////////////////////////////////////
    // explicit instantiation for the correct archive types
    template HPX_COMPONENT_EXPORT void
    serialize(input_archive&, interpolate1d::dimension&, unsigned int const);
    template HPX_COMPONENT_EXPORT void
    serialize(output_archive&, interpolate1d::dimension&, unsigned int const);
}}


