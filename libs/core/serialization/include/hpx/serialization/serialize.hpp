//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2014-2015 Anton Bikineev
//  Copyright (c) 2022-2025 Hartmut Kaiser
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

    HPX_CXX_CORE_EXPORT template <typename T>
    HPX_FORCEINLINE output_archive& operator<<(output_archive& ar, T const& t)
    {
        ar.save(t);
        return ar;
    }

    HPX_CXX_CORE_EXPORT template <typename T>
    HPX_FORCEINLINE input_archive& operator>>(input_archive& ar, T& t)
    {
        ar.load(t);
        return ar;
    }

    HPX_CXX_CORE_EXPORT template <typename T>
    HPX_FORCEINLINE output_archive& operator&(    //-V524
        output_archive& ar, T const& t)
    {
        ar.save(t);
        return ar;
    }

    HPX_CXX_CORE_EXPORT template <typename T>
    HPX_FORCEINLINE input_archive& operator&(input_archive& ar, T& t)    //-V524
    {
        ar.load(t);
        return ar;
    }

    namespace detail {

        HPX_CXX_CORE_EXPORT template <typename T>
        void serialize_one(output_archive& ar, T const& t)
        {
            ar.save(t);
        }

        HPX_CXX_CORE_EXPORT template <typename T>
        void serialize_one(input_archive& ar, T& t)
        {
            ar.load(t);
        }
    }    // namespace detail
}    // namespace hpx::serialization
