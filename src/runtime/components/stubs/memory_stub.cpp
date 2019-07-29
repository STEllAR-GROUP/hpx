//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/apply.hpp>
#include <hpx/assertion.hpp>
#include <hpx/async.hpp>
#include <hpx/runtime/components/stubs/memory.hpp>

#include <cstddef>
#include <cstdint>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    hpx::future<naming::id_type> memory::allocate(hpx::id_type const& id,
        std::size_t size)
    {
        return hpx::async<components::server::allocate_action>(id, size);
    }
    naming::id_type memory::allocate(launch::sync_policy,
        hpx::id_type const& id, std::size_t size, error_code& ec)
    {
        return allocate(id, size).get(ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    void memory::store8(launch::apply_policy, hpx::id_type const& id,
        std::uint8_t value)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        hpx::apply<components::server::memory::store8_action>(
            id, id.get_lsb(), value);
    }
    hpx::future<void> memory::store8(hpx::id_type const& id,
        std::uint8_t value)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        return hpx::async<components::server::memory::store8_action>(
            id, id.get_lsb(), value);
    }
    void memory::store8(launch::sync_policy, hpx::id_type const& id,
        std::uint8_t value, error_code& ec)
    {
        store8(id, value).get(ec);
    }

    void memory::store16(launch::apply_policy, hpx::id_type const& id,
        std::uint16_t value)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        hpx::apply<components::server::memory::store16_action>(
            id, id.get_lsb(), value);
    }
    hpx::future<void> memory::store16(hpx::id_type const& id,
        std::uint16_t value)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        return hpx::async<components::server::memory::store16_action>(
            id, id.get_lsb(), value);
    }
    void memory::store16(launch::sync_policy, hpx::id_type const& id,
        std::uint16_t value, error_code& ec)
    {
        store16(id, value).get(ec);
    }

    void memory::store32(launch::apply_policy, hpx::id_type const& id,
        std::uint32_t value)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        hpx::apply<components::server::memory::store32_action>(
            id, id.get_lsb(), value);
    }
    hpx::future<void> memory::store32(hpx::id_type const& id,
        std::uint32_t value)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        return hpx::async<components::server::memory::store32_action>(
            id, id.get_lsb(), value);
    }
    void memory::store32(launch::sync_policy, hpx::id_type const& id,
        std::uint32_t value, error_code& ec)
    {
        store32(id, value).get(ec);
    }

    void memory::store64(launch::apply_policy, hpx::id_type const& id,
        std::uint64_t value)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        hpx::apply<components::server::memory::store64_action>(
            id, id.get_lsb(), value);
    }
    hpx::future<void> memory::store64(hpx::id_type const& id,
        std::uint64_t value)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        return hpx::async<components::server::memory::store64_action>(
            id, id.get_lsb(), value);
    }
    void memory::store64(launch::sync_policy, hpx::id_type const& id,
        std::uint64_t value, error_code& ec)
    {
        store64(id, value).get(ec);
    }

    void memory::store128(launch::apply_policy, hpx::id_type const& id,
        components::server::memory::uint128_t const& value)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        hpx::apply<components::server::memory::store128_action>(
            id, id.get_lsb(), value);
    }
    hpx::future<void> memory::store128(hpx::id_type const& id,
        components::server::memory::uint128_t const& value)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        return hpx::async<components::server::memory::store128_action>(
            id, id.get_lsb(), value);
    }
    void memory::store128(launch::sync_policy, hpx::id_type const& id,
        components::server::memory::uint128_t const& value, error_code& ec)
    {
        store128(id, value).get(ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<std::uint8_t> memory::load8(hpx::id_type const& id)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        return hpx::async<components::server::memory::load8_action>(
            id, id.get_lsb());
    }
    std::uint8_t memory::load8(launch::sync_policy,
        hpx::id_type const& id, error_code& ec)
    {
        return load8(id).get(ec);
    }

    hpx::future<std::uint16_t> memory::load16(hpx::id_type const& id)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        return hpx::async<components::server::memory::load16_action>(
            id, id.get_lsb());
    }
    std::uint16_t memory::load16(launch::sync_policy,
        hpx::id_type const& id, error_code& ec)
    {
        return load16(id).get(ec);
    }

    hpx::future<std::uint32_t> memory::load32(hpx::id_type const& id)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        return hpx::async<components::server::memory::load32_action>(
            id, id.get_lsb());
    }
    std::uint32_t memory::load32(launch::sync_policy,
        hpx::id_type const& id, error_code& ec)
    {
        return load32(id).get(ec);
    }

    hpx::future<std::uint64_t> memory::load64(hpx::id_type const& id)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        return hpx::async<components::server::memory::load64_action>(
            id, id.get_lsb());
    }
    std::uint64_t memory::load64(launch::sync_policy,
        hpx::id_type const& id, error_code& ec)
    {
        return load64(id).get(ec);
    }

    hpx::future<components::server::memory::uint128_t> memory::load128(
        hpx::id_type const& id)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        return hpx::async<components::server::memory::load128_action>(
            id, id.get_lsb());
    }
    components::server::memory::uint128_t memory::load128(launch::sync_policy,
        hpx::id_type const& id, error_code& ec)
    {
        return load128(id).get(ec);
    }
}}}
