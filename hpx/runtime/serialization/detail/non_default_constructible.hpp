//  Copyright (c) 2017 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_NON_DEFAULT_CONSTRUCTIBLE_HPP
#define HPX_SERIALIZATION_NON_DEFAULT_CONSTRUCTIBLE_HPP

#include <hpx/config.hpp>

#include <memory>

namespace hpx { namespace serialization { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // default fall-backs for serializing non-default-constructible types
    template <typename Archive, typename T>
    void save_construct_data(Archive&, T const*, unsigned)
    {
    }

    // by default fall back to in-place default construction
    template <typename Archive, typename T>
    void load_construct_data(Archive& ar, T* t, unsigned)
    {
        ::new (t) T;
    }
}}}

#endif
