//  Copyright (c) 2025 Ujjwal Shekhar
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/serialization/config/defines.hpp>
#include <hpx/serialization/brace_initializable_fwd.hpp>

#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/traits/polymorphic_traits.hpp>

#include <hpx/serialization/base_object.hpp>

// This file is only ever included by access.hpp
// but we will still guard against direct inclusion
#if defined(HPX_HAVE_CXX26_EXPERIMENTAL_META) && defined(HPX_SERIALIZATION_ALLOW_AUTO_GENERATE)
#include <experimental/meta>

namespace hpx::serialization::detail {
    // This function uses C++26 reflection capabilities to generate
    // serialization functions for types that don't have them already.
    HPX_CXX_EXPORT template <typename Archive, typename T>
    void refl_serialize(Archive& ar, T& t, unsigned /*version*/)
    {
        constexpr auto ctx = std::meta::access_context::unchecked();

        // Serialize all bases
        template for (constexpr auto base_info : std::meta::bases_of(^^T, ctx)) {
            using BaseType = typename[:std::meta::type_of(base_info):];
            ar & hpx::serialization::base_object<BaseType>(t);
        }

        // Serialize all members
        template for (
            constexpr auto member : std::meta::nonstatic_data_members_of(^^T, ctx))
        {
            using MemberType = typename[:std::meta::type_of(member):];

            // Since we are using an unchecked context, this might
            // be a private/protected member, we will have to do manual
            // pointer arithmetic to get the member and codegen the
            // archive handling logic
            // TODO: If intrusion given, we can directly access the member
            if constexpr (std::meta::is_public(member)) {
                ar & t.[:member:];
            } else {
                constexpr size_t offset = std::meta::offset_of(member);
                ar & *static_cast<MemberType*>(
                        reinterpret_cast<char*>(&t) + offset);
            }
        }
    }
}    // namespace hpx::serialization::detail

#endif
