//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_STUBS_MEMORY_MAR_06_2014_0831PM)
#define HPX_COMPONENTS_STUBS_MEMORY_MAR_06_2014_0831PM

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/server/memory.hpp>

namespace hpx { namespace components { namespace stubs
{
    struct HPX_EXPORT memory
    {
        ///////////////////////////////////////////////////////////////////////
        static hpx::future<naming::id_type> allocate(
            hpx::id_type const& id, std::size_t size);
        static naming::id_type allocate_sync(hpx::id_type const& id,
            std::size_t size, error_code& ec = throws);

        ///////////////////////////////////////////////////////////////////////
        static void store8_async(hpx::id_type const& id, boost::uint8_t value);
        static hpx::future<void> store8(hpx::id_type const& id,
            boost::uint8_t value);
        static void store8_sync(hpx::id_type const& id, boost::uint8_t value,
            error_code& ec = throws);

        static void store16_async(hpx::id_type const& id, boost::uint16_t value);
        static hpx::future<void> store16(hpx::id_type const& id,
            boost::uint16_t value);
        static void store16_sync(hpx::id_type const& id, boost::uint16_t value,
            error_code& ec = throws);

        static void store32_async(hpx::id_type const& id, boost::uint32_t value);
        static hpx::future<void> store32(hpx::id_type const& id,
            boost::uint32_t value);
        static void store32_sync(hpx::id_type const& id, boost::uint32_t value,
            error_code& ec = throws);

        static void store64_async(hpx::id_type const& id, boost::uint64_t value);
        static hpx::future<void> store64(hpx::id_type const& id,
            boost::uint64_t value);
        static void store64_sync(hpx::id_type const& id, boost::uint64_t value,
            error_code& ec = throws);

        static void store128_async(hpx::id_type const& id,
            components::server::memory::uint128_t const& value);
        static hpx::future<void> store128(hpx::id_type const& id,
            components::server::memory::uint128_t const& value);
        static void store128_sync(hpx::id_type const& id,
            components::server::memory::uint128_t const& value,
            error_code& ec = throws);

        ///////////////////////////////////////////////////////////////////////
        static hpx::future<boost::uint8_t> load8(hpx::id_type const& id);
        static boost::uint8_t load8_sync(hpx::id_type const& id,
            error_code& ec = throws);

        static hpx::future<boost::uint16_t> load16(hpx::id_type const& id);
        static boost::uint16_t load16_sync(hpx::id_type const& id,
            error_code& ec = throws);

        static hpx::future<boost::uint32_t> load32(hpx::id_type const& id);
        static boost::uint32_t load32_sync(hpx::id_type const& id,
            error_code& ec = throws);

        static hpx::future<boost::uint64_t> load64(hpx::id_type const& id);
        static boost::uint64_t load64_sync(hpx::id_type const& id,
            error_code& ec = throws);

        static hpx::future<components::server::memory::uint128_t> load128(
            hpx::id_type const& id);
        static components::server::memory::uint128_t load128_sync(hpx::id_type const& id,
            error_code& ec = throws);
    };
}}}

#endif
