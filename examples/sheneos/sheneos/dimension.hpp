//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SHENEOS_DIMENSION3D_AUG_08_2011_1222PM)
#define HPX_SHENEOS_DIMENSION3D_AUG_08_2011_1222PM

#include <hpx/hpx_fwd.hpp>

#include <boost/serialization/serialization.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace sheneos 
{
    struct dimension
    {
        enum {
            ye = 0, temp = 1, rho = 2,
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
    void serialize(Archive&, sheneos::dimension&, unsigned int const);
}}

#endif


