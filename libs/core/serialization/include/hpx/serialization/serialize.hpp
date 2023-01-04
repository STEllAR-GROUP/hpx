//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2014-2015 Anton Bikineev
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/access.hpp>
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>

namespace hpx::serialization {

    template <typename T>
    HPX_FORCEINLINE output_archive& operator<<(output_archive& ar, T const& t)
    {
        ar.save(t);
        return ar;
    }

    template <typename T>
    HPX_FORCEINLINE input_archive& operator>>(input_archive& ar, T& t)
    {
        ar.load(t);
        return ar;
    }

    template <typename T>
    HPX_FORCEINLINE output_archive& operator&(    //-V524
        output_archive& ar, T const& t)
    {
        ar.save(t);
        return ar;
    }

    template <typename T>
    HPX_FORCEINLINE input_archive& operator&(input_archive& ar, T& t)    //-V524
    {
        ar.load(t);
        return ar;
    }

    namespace detail {

        template <typename Archive, typename T>
        void serialize_one(Archive& ar, T& t)
        {
            // clang-format off
            ar & t;
            // clang-format on
        }
    }    // namespace detail
}    // namespace hpx::serialization

#include <hpx/serialization/detail/polymorphic_nonintrusive_factory_impl.hpp>
