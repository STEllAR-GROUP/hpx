///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/config.hpp>

#include <hpx/compute/host/target.hpp>
#include <hpx/compute/traits/access_target.hpp>

#include <vector>

namespace hpx { namespace compute { namespace traits {
    template <>
    struct access_target<host::target>
    {
        typedef host::target target_type;

        template <typename T>
        static T const& read(target_type const& /* tgt */, T const* t)
        {
            return *t;
        }

        template <typename T>
        static void write(target_type const& /* tgt */, T* dst, T const* src)
        {
            *dst = *src;
        }
    };

    template <>
    struct access_target<std::vector<host::target>>
    {
        typedef std::vector<host::target> target_type;

        template <typename T>
        static T const& read(target_type const& /* tgt */, T const* t)
        {
            return *t;
        }

        template <typename T>
        static void write(target_type const& /* tgt */, T* dst, T const* src)
        {
            *dst = *src;
        }
    };
}}}    // namespace hpx::compute::traits
