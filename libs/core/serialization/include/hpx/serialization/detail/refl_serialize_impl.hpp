//  Copyright (c) 2026 Ujjwal Shekhar
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>

#include <hpx/serialization/base_object.hpp>

#include <iostream>

// This file is only ever included by access.hpp
// but we will still guard against direct inclusion
#if defined(HPX_HAVE_CXX26_EXPERIMENTAL_META) &&                               \
    defined(HPX_SERIALIZATION_HAVE_ALLOW_AUTO_GENERATE)
#include <experimental/meta>

namespace hpx::serialization::detail {
    HPX_CXX_EXPORT template <typename Archive, typename T>
    void refl_serialize(Archive& ar, T& t, unsigned /*version*/)
    {
        constexpr auto ctx = std::meta::access_context::unchecked();

        // Serialize all bases
        template for (constexpr auto base_info :
            std::define_static_array(std::meta::bases_of(^^T, ctx)))
        {
            if constexpr (std::meta::is_public(base_info))
            {
                using BaseType = typename[:std::meta::type_of(base_info):];
                ar& hpx::serialization::base_object<BaseType>(t);
            }
        }

        // Serialize all members
        template for (constexpr auto member : std::define_static_array(
                          std::meta::nonstatic_data_members_of(^^T, ctx)))
        {
            // Since we are using an unchecked context, this might
            // be a private/protected member, we will have to do manual
            // pointer arithmetic to get the member and codegen the
            // archive handling logic

            // TODO: If intrusion provided, we can directly access the member
            // so we should try figuring out the access context that we have
            // and optimize for that case
            if constexpr (std::meta::is_public(member))
            {
                using MemberType = typename[:std::meta::type_of(member):];
                ar& const_cast<MemberType&>(t.[:member:]);
            }
            else
            {
                constexpr auto offset_info = std::meta::offset_of(member);
                static_assert(offset_info.bits == 0,
                    "Reflection serialization does not support bitfields");

                using MemberType = typename[:std::meta::type_of(member):];
                constexpr size_t offset = offset_info.bytes;

                if constexpr (std::is_const_v<T>)
                {    // save path
                    auto* member_ptr = reinterpret_cast<MemberType const*>(
                        reinterpret_cast<std::byte const*>(std::addressof(t)) +
                        offset);
                    ar&* member_ptr;
                }
                else
                {    // load path
                    std::byte* base_addr =
                        reinterpret_cast<std::byte*>(std::addressof(t));
                    std::byte* target_addr = base_addr + offset;

                    auto* member_ptr = reinterpret_cast<MemberType*>(
                        reinterpret_cast<std::byte*>(std::addressof(t)) +
                        offset);
                    ar&* member_ptr;
                }
            }
        }
    }
}    // namespace hpx::serialization::detail

#endif
