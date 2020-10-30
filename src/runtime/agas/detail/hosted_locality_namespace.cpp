////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2012-2017 Hartmut Kaiser
//  Copyright (c) 2016 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/assert.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/agas/detail/hosted_locality_namespace.hpp>
#include <hpx/runtime/agas/server/locality_namespace.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/serialization/vector.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace hpx { namespace agas { namespace detail
{
    hosted_locality_namespace::hosted_locality_namespace(naming::address addr)
      : gid_(
            naming::gid_type(HPX_AGAS_LOCALITY_NS_MSB, HPX_AGAS_LOCALITY_NS_LSB),
            naming::id_type::unmanaged)
      , addr_(addr)
    {}

    std::uint32_t hosted_locality_namespace::allocate(
        parcelset::endpoints_type const& /* endpoints */,
        std::uint64_t /* count */, std::uint32_t /* num_threads */,
        naming::gid_type const& /* suggested_prefix */)
    {
        return 0;
    }

    void hosted_locality_namespace::free(naming::gid_type const& locality)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        server::locality_namespace::free_action action;
        action(gid_, locality);
#else
        HPX_UNUSED(locality);
        HPX_ASSERT(false);
#endif
    }

    std::vector<std::uint32_t> hosted_locality_namespace::localities()
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        server::locality_namespace::localities_action action;
        return action(gid_);
#else
        HPX_ASSERT(false);
        return std::vector<std::uint32_t>{};
#endif
    }

    parcelset::endpoints_type hosted_locality_namespace::resolve_locality(
        naming::gid_type const& locality)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        server::locality_namespace::resolve_locality_action action;
        future<parcelset::endpoints_type> endpoints_future =
            hpx::async(action, gid_, locality);

        if (nullptr == threads::get_self_ptr())
        {
            // this should happen only during bootstrap
            HPX_ASSERT(hpx::is_starting());

            while(!endpoints_future.is_ready())
                /**/;
        }

        return endpoints_future.get();
#else
        HPX_UNUSED(locality);
        HPX_ASSERT(false);
        return parcelset::endpoints_type{};
#endif
    }

    std::uint32_t hosted_locality_namespace::get_num_localities()
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        server::locality_namespace::get_num_localities_action action;
        return action(gid_);
#else
        HPX_ASSERT(false);
        return std::uint32_t{};
#endif
    }

    hpx::future<std::uint32_t> hosted_locality_namespace::get_num_localities_async()
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        server::locality_namespace::get_num_localities_action action;
        return hpx::async(action, gid_);
#else
        HPX_ASSERT(false);
        return hpx::make_ready_future(std::uint32_t{});
#endif
    }

    std::vector<std::uint32_t> hosted_locality_namespace::get_num_threads()
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        server::locality_namespace::get_num_threads_action action;
        return action(gid_);
#else
        HPX_ASSERT(false);
        return std::vector<std::uint32_t>{};
#endif
    }

    hpx::future<std::vector<std::uint32_t>>
    hosted_locality_namespace::get_num_threads_async()
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        server::locality_namespace::get_num_threads_action action;
        return hpx::async(action, gid_);
#else
        HPX_ASSERT(false);
        return hpx::make_ready_future(std::vector<std::uint32_t>{});
#endif
    }

    std::uint32_t hosted_locality_namespace::get_num_overall_threads()
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        server::locality_namespace::get_num_overall_threads_action action;
        return action(gid_);
#else
        HPX_ASSERT(false);
        return hpx::resource::get_num_threads();
#endif
    }

    hpx::future<std::uint32_t>
    hosted_locality_namespace::get_num_overall_threads_async()
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        server::locality_namespace::get_num_overall_threads_action action;
        return hpx::async(action, gid_);
#else
        HPX_ASSERT(false);
        return hpx::make_ready_future(std::uint32_t{});
#endif
    }

    naming::gid_type hosted_locality_namespace::statistics_counter(std::string name)
    {
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        server::locality_namespace::statistics_counter_action action;
        return action(gid_, name).get_gid();
#else
        HPX_UNUSED(name);
        HPX_ASSERT(false);
        return naming::gid_type{};
#endif
    }
}}}

#endif
