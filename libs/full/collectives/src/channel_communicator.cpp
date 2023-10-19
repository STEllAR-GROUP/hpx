//  Copyright (c) 2020-2021 Hartmut Kaiser
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
#include <hpx/runtime_components/new.hpp>

#include <cstddef>
#include <memory>
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
}    // namespace hpx::collectives

#endif    // !HPX_COMPUTE_DEVICE_CODE
