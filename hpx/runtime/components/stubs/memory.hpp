//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_MEMORY_MAR_06_2014_0831PM)
#define HPX_COMPONENTS_STUBS_MEMORY_MAR_06_2014_0831PM

#include <hpx/config.hpp>
#include <hpx/errors.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/server/memory.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming/name.hpp>

#include <cstddef>
#include <cstdint>

namespace hpx { namespace components { namespace stubs
{
    struct HPX_EXPORT memory
    {
        ///////////////////////////////////////////////////////////////////////
        static hpx::future<naming::id_type> allocate(
            id_type const& id, std::size_t size);
        static naming::id_type allocate(launch::sync_policy,
            id_type const& id, std::size_t size, error_code& ec = throws);

        ///////////////////////////////////////////////////////////////////////
        static hpx::future<void> store8(id_type const& id, std::uint8_t value);
        static void store8(launch::apply_policy,
            id_type const& id, std::uint8_t value);
        static void store8(launch::sync_policy,
            id_type const& id, std::uint8_t value, error_code& ec = throws);

        static hpx::future<void> store16(id_type const& id,
            std::uint16_t value);
        static void store16(launch::apply_policy,
            id_type const& id, std::uint16_t value);
        static void store16(launch::sync_policy,
            id_type const& id, std::uint16_t value, error_code& ec = throws);

        static hpx::future<void> store32(id_type const& id,
            std::uint32_t value);
        static void store32(launch::apply_policy,
            id_type const& id, std::uint32_t value);
        static void store32(launch::sync_policy,
            id_type const& id, std::uint32_t value, error_code& ec = throws);

        static hpx::future<void> store64(id_type const& id,
            std::uint64_t value);
        static void store64(launch::apply_policy,
            id_type const& id, std::uint64_t value);
        static void store64(launch::sync_policy,
            id_type const& id, std::uint64_t value,
            error_code& ec = throws);

        static hpx::future<void> store128(id_type const& id,
            components::server::memory::uint128_t const& value);
        static void store128(launch::apply_policy, id_type const& id,
            components::server::memory::uint128_t const& value);
        static void store128(launch::sync_policy,
            id_type const& id,
            components::server::memory::uint128_t const& value,
            error_code& ec = throws);

        ///////////////////////////////////////////////////////////////////////
        static hpx::future<std::uint8_t> load8(id_type const& id);
        static std::uint8_t load8(launch::sync_policy, id_type const& id,
            error_code& ec = throws);

        static hpx::future<std::uint16_t> load16(id_type const& id);
        static std::uint16_t load16(launch::sync_policy, id_type const& id,
            error_code& ec = throws);

        static hpx::future<std::uint32_t> load32(id_type const& id);
        static std::uint32_t load32(launch::sync_policy, id_type const& id,
            error_code& ec = throws);

        static hpx::future<std::uint64_t> load64(id_type const& id);
        static std::uint64_t load64(launch::sync_policy, id_type const& id,
            error_code& ec = throws);

        static hpx::future<components::server::memory::uint128_t> load128(
            id_type const& id);
        static components::server::memory::uint128_t
        load128(launch::sync_policy, id_type const& id, error_code& ec = throws);
    };
}}}

#endif
