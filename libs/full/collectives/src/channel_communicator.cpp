//  Copyright (c) 2020-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/collectives/channel_communicator.hpp>
#include <hpx/components/basename_registration.hpp>
#include <hpx/components/client.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/components_base/server/component.hpp>
#include <hpx/errors/exception.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/lock_registration.hpp>
#include <hpx/runtime_components/new.hpp>
#include <hpx/synchronization/mutex.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::collectives {

    ///////////////////////////////////////////////////////////////////////////
    channel_communicator::channel_communicator() = default;
    channel_communicator::~channel_communicator() = default;

    channel_communicator::channel_communicator(char const* basename,
        num_sites_arg num_sites, this_site_arg this_site,
        components::client<detail::channel_communicator_server>&& here)
      : comm_(std::make_shared<detail::channel_communicator>(
            basename, num_sites.argument_, this_site.argument_, HPX_MOVE(here)))
    {
    }

    void channel_communicator::free()
    {
        comm_.reset();
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<channel_communicator> create_channel_communicator(
        char const* basename, num_sites_arg num_sites, this_site_arg this_site)
    {
        if (num_sites == static_cast<std::size_t>(-1))
        {
            num_sites = agas::get_num_localities(hpx::launch::sync);
        }
        if (this_site == static_cast<std::size_t>(-1))
        {
            this_site = agas::get_locality_id();
        }

        HPX_ASSERT(this_site < num_sites);

        using client_type =
            hpx::components::client<detail::channel_communicator_server>;

        // create a new communicator on each locality
        client_type c = hpx::local_new<client_type>(num_sites.argument_);

        // register the communicator's id using the given basename,
        // this keeps the communicator alive
        auto f = c.register_as(
            hpx::detail::name_from_basename(basename, this_site.argument_));

        return f.then(hpx::launch::sync,
            [=, target = HPX_MOVE(c)](hpx::future<bool>&& fut) mutable {
                if (bool const result = fut.get(); !result)
                {
                    HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                        "hpx::collectives::detail::create_channel_communicator",
                        "the given base name for the communicator operation "
                        "was already registered: {}",
                        target.registered_name());
                }
                return channel_communicator(
                    basename, num_sites, this_site, HPX_MOVE(target));
            });
    }

    channel_communicator create_channel_communicator(hpx::launch::sync_policy,
        char const* basename, num_sites_arg num_sites, this_site_arg this_site)
    {
        return create_channel_communicator(basename, num_sites, this_site)
            .get();
    }

    ///////////////////////////////////////////////////////////////////////////
    // Predefined channel (p2p) communicator
    namespace {

        channel_communicator world_channel_communicator;
        hpx::mutex world_channel_communicator_mtx;
    }    // namespace

    channel_communicator get_world_channel_communicator()
    {
        detail::create_world_channel_communicator();
        return world_channel_communicator;
    }

    namespace detail {

        void create_world_channel_communicator()
        {
            std::unique_lock<hpx::mutex> l(world_channel_communicator_mtx);
            [[maybe_unused]] util::ignore_while_checking il(&l);

            if (!world_channel_communicator)
            {
                auto const num_sites =
                    num_sites_arg(agas::get_num_localities(hpx::launch::sync));
                auto const this_site = this_site_arg(agas::get_locality_id());

                world_channel_communicator =
                    collectives::create_channel_communicator(hpx::launch::sync,
                        "world_channel_communicator", num_sites, this_site);
            }
        }

        void reset_world_channel_communicator()
        {
            if (world_channel_communicator)
            {
                world_channel_communicator.free();
            }
        }
    }    // namespace detail
}    // namespace hpx::collectives

#endif    // !HPX_COMPUTE_DEVICE_CODE
