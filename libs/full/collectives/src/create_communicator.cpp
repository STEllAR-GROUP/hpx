//  Copyright (c) 2020-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/assert.hpp>
#include <hpx/collectives/create_communicator.hpp>
#include <hpx/components/basename_registration.hpp>
#include <hpx/components/client.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/components_base/server/component.hpp>
#include <hpx/errors/exception.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/runtime_components/component_factory.hpp>
#include <hpx/runtime_components/new.hpp>
#include <hpx/runtime_distributed/server/runtime_support.hpp>

#include <cstddef>
#include <string>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
using collectives_component =
    hpx::components::component<hpx::collectives::detail::communicator_server>;

HPX_REGISTER_COMPONENT(collectives_component);

namespace hpx { namespace collectives {

    ///////////////////////////////////////////////////////////////////////////
    communicator create_communicator(char const* basename,
        num_sites_arg num_sites, this_site_arg this_site,
        generation_arg generation, root_site_arg root_site)
    {
        if (num_sites == std::size_t(-1))
        {
            num_sites = static_cast<std::size_t>(
                agas::get_num_localities(hpx::launch::sync));
        }
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(agas::get_locality_id());
            if (root_site == std::size_t(-1))
            {
                root_site = this_site;
            }
        }

        HPX_ASSERT(this_site < num_sites);
        HPX_ASSERT(root_site != std::size_t(-1) && root_site < num_sites);

        std::string name(basename);
        if (generation != std::size_t(-1))
        {
            name += std::to_string(generation) + "/";
        }

        if (this_site == root_site)
        {
            // create a new communicator
            communicator c = hpx::local_new<communicator>(num_sites);

            // register the communicator's id using the given basename,
            // this keeps the communicator alive
            auto f = c.register_as(
                hpx::detail::name_from_basename(std::move(name), this_site));

            return f.then(hpx::launch::sync,
                [=, target = std::move(c)](hpx::future<bool>&& f) mutable {
                    bool result = f.get();
                    if (!result)
                    {
                        HPX_THROW_EXCEPTION(bad_parameter,
                            "hpx::collectives::detail::create_communicator",
                            hpx::util::format(
                                "the given base name for the communicator "
                                "operation was already registered: {}",
                                target.registered_name()));
                    }
                    target.set_info(num_sites, this_site);
                    return target;
                });
        }

        // find existing communicator
        return hpx::find_from_basename<communicator>(
            std::move(name), root_site);
    }
}}    // namespace hpx::collectives

#endif    // !HPX_COMPUTE_DEVICE_CODE
