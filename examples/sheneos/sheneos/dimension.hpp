//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SHENEOS_DIMENSION3D_AUG_08_2011_1222PM)
#define HPX_SHENEOS_DIMENSION3D_AUG_08_2011_1222PM

#include <hpx/hpx_fwd.hpp>

#include <hpx/runtime/serialization/serialize.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace sheneos
{
    struct dimension
    {
        enum type {
            ye = 0,
            temp = 1,
            rho = 2,
            dim = 3
        };

        dimension() : offset_(0), count_(0), size_(0) {}
        dimension(std::size_t size) : offset_(0), count_(0), size_(size) {}

        std::size_t offset_;    ///< Dimension offset.
        std::size_t count_;     ///< Dimension count.
        std::size_t size_;      ///< Overall size of this dimension.
    };
}

///////////////////////////////////////////////////////////////////////////////
// Non-intrusive serialization support.
namespace hpx { namespace serialization
{
    HPX_COMPONENT_EXPORT void
    serialize(input_archive& ar, sheneos::dimension& dim,
        unsigned int const);

    HPX_COMPONENT_EXPORT void
    serialize(output_archive& ar, sheneos::dimension& dim,
        unsigned int const);
}}

#endif

