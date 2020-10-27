//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/collectives/detail/communicator.hpp>
#include <hpx/exception.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/basename_registration.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/new.hpp>
#include <hpx/runtime/components/server/component.hpp>
#include <hpx/runtime/components/server/runtime_support.hpp>

#include <cstddef>
#include <string>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
using collectives_component =
    hpx::components::component<hpx::lcos::detail::communicator_server>;

HPX_REGISTER_COMPONENT(collectives_component);

namespace hpx { namespace lcos { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<hpx::id_type> register_communicator_name(
        hpx::future<hpx::id_type>&& f, std::string basename, std::size_t site)
    {
        hpx::id_type target = f.get();

        // Register unmanaged id to avoid cyclic dependencies, unregister
        // is done after all data has been collected in the component above.
        hpx::future<bool> result =
            hpx::register_with_basename(basename, target, site);

        return result.then(hpx::launch::sync,
            [target = std::move(target), basename = std::move(basename)](
                hpx::future<bool>&& f) -> hpx::id_type {
                bool result = f.get();
                if (!result)
                {
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "hpx::lcos::detail::register_communicator_name",
                        "the given base name for the communicator "
                        "operation was already registered: " +
                            basename);
                }
                return target;
            });
    }

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<hpx::id_type> create_communicator(char const* basename,
        std::size_t num_sites, std::size_t generation, std::size_t this_site,
        std::size_t num_values)
    {
        if (num_sites == std::size_t(-1))
        {
            num_sites = static_cast<std::size_t>(
                hpx::get_num_localities(hpx::launch::sync));
        }
        if (this_site == std::size_t(-1))
        {
            this_site = static_cast<std::size_t>(hpx::get_locality_id());
        }
        if (num_values == std::size_t(-1))
        {
            num_values = num_sites;
        }

        std::string name(basename);
        if (generation != std::size_t(-1))
        {
            name += std::to_string(generation) + "/";
        }

        // create a new communicator_server
        hpx::future<hpx::id_type> id = hpx::new_<detail::communicator_server>(
            hpx::find_here(), num_sites, name, this_site, num_values);

        // register the communicator's id using the given basename
        return id.then(hpx::launch::sync,
            util::bind_back(&detail::register_communicator_name,
                std::move(name), this_site));
    }
}}}    // namespace hpx::lcos::detail

#endif
