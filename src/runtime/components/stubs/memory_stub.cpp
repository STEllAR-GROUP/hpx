//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/async.hpp>
#include <hpx/apply.hpp>
#include <hpx/runtime/components/stubs/memory.hpp>

namespace hpx { namespace components { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    hpx::future<naming::id_type> memory::allocate(hpx::id_type const& id,
        std::size_t size)
    {
        return hpx::async<components::server::allocate_action>(id, size);
    }
    naming::id_type memory::allocate_sync(hpx::id_type const& id,
        std::size_t size, error_code& ec)
    {
        return allocate(id, size).get(ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    void memory::store8_async(hpx::id_type const& id,
        boost::uint8_t value)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        hpx::apply<components::server::memory::store8_action>(
            id, id.get_lsb(), value);
    }
    hpx::future<void> memory::store8(hpx::id_type const& id,
        boost::uint8_t value)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        return hpx::async<components::server::memory::store8_action>(
            id, id.get_lsb(), value);
    }
    void memory::store8_sync(hpx::id_type const& id, boost::uint8_t value,
        error_code& ec)
    {
        store8(id, value).get(ec);
    }

    void memory::store16_async(hpx::id_type const& id,
        boost::uint16_t value)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        hpx::apply<components::server::memory::store16_action>(
            id, id.get_lsb(), value);
    }
    hpx::future<void> memory::store16(hpx::id_type const& id,
        boost::uint16_t value)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        return hpx::async<components::server::memory::store16_action>(
            id, id.get_lsb(), value);
    }
    void memory::store16_sync(hpx::id_type const& id, boost::uint16_t value,
        error_code& ec)
    {
        store16(id, value).get(ec);
    }

    void memory::store32_async(hpx::id_type const& id,
        boost::uint32_t value)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        hpx::apply<components::server::memory::store32_action>(
            id, id.get_lsb(), value);
    }
    hpx::future<void> memory::store32(hpx::id_type const& id,
        boost::uint32_t value)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        return hpx::async<components::server::memory::store32_action>(
            id, id.get_lsb(), value);
    }
    void memory::store32_sync(hpx::id_type const& id, boost::uint32_t value,
        error_code& ec)
    {
        store32(id, value).get(ec);
    }

    void memory::store64_async(hpx::id_type const& id,
        boost::uint64_t value)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        hpx::apply<components::server::memory::store64_action>(
            id, id.get_lsb(), value);
    }
    hpx::future<void> memory::store64(hpx::id_type const& id,
        boost::uint64_t value)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        return hpx::async<components::server::memory::store64_action>(
            id, id.get_lsb(), value);
    }
    void memory::store64_sync(hpx::id_type const& id, boost::uint64_t value,
        error_code& ec)
    {
        store64(id, value).get(ec);
    }

    void memory::store128_async(hpx::id_type const& id,
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
    void memory::store128_sync(hpx::id_type const& id,
        components::server::memory::uint128_t const& value, error_code& ec)
    {
        store128(id, value).get(ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<boost::uint8_t> memory::load8(hpx::id_type const& id)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        return hpx::async<components::server::memory::load8_action>(
            id, id.get_lsb());
    }
    boost::uint8_t memory::load8_sync(hpx::id_type const& id, error_code& ec)
    {
        return load8(id).get(ec);
    }

    hpx::future<boost::uint16_t> memory::load16(hpx::id_type const& id)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        return hpx::async<components::server::memory::load16_action>(
            id, id.get_lsb());
    }
    boost::uint16_t memory::load16_sync(hpx::id_type const& id, error_code& ec)
    {
        return load16(id).get(ec);
    }

    hpx::future<boost::uint32_t> memory::load32(hpx::id_type const& id)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        return hpx::async<components::server::memory::load32_action>(
            id, id.get_lsb());
    }
    boost::uint32_t memory::load32_sync(hpx::id_type const& id, error_code& ec)
    {
        return load32(id).get(ec);
    }

    hpx::future<boost::uint64_t> memory::load64(hpx::id_type const& id)
    {
        HPX_ASSERT(naming::refers_to_virtual_memory(id.get_gid()));
        return hpx::async<components::server::memory::load64_action>(
            id, id.get_lsb());
    }
    boost::uint64_t memory::load64_sync(hpx::id_type const& id, error_code& ec)
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
    components::server::memory::uint128_t memory::load128_sync(hpx::id_type const& id,
        error_code& ec)
    {
        return load128(id).get(ec);
    }
}}}
