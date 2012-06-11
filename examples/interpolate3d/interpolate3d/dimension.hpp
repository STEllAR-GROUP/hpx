//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_DIMENSION3D_AUG_07_2011_1143AM)
#define HPX_DIMENSION3D_AUG_07_2011_1143AM

#include <hpx/hpx_fwd.hpp>

#include <boost/serialization/serialization.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace interpolate3d
{
    struct dimension
    {
        enum {
            x = 0, y = 1, z = 2,
            dim = 3
        };

        dimension() : offset_(0), count_(0), size_(0) {}
        dimension(std::size_t size) : offset_(0), count_(0), size_(size) {}

        std::size_t offset_;    // lower index
        std::size_t count_;     // upper index
        std::size_t size_;      // overall size of this dimension
    };
}

///////////////////////////////////////////////////////////////////////////////
// non-intrusive serialization
namespace boost { namespace serialization
{
    template <typename Archive>
    void serialize(Archive&, interpolate3d::dimension&, unsigned int const);
}}

#endif


