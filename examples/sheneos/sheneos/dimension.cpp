//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/include/serialization.hpp>

#include "dimension.hpp"

namespace hpx { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    // Implementation of the serialization function.
    void serialize(input_archive& ar, sheneos::dimension& dim, unsigned int const)
    {
        ar & dim.offset_ & dim.count_ & dim.size_;
    }

    void serialize(output_archive& ar, sheneos::dimension& dim, unsigned int const)
    {
        ar & dim.offset_ & dim.count_ & dim.size_;
    }
}}

