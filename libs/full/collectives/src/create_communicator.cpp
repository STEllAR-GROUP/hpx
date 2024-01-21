//  Copyright (c) 2020-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/collectives/create_communicator.hpp>
#include <hpx/components/basename_registration.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/components_base/server/component.hpp>
#include <hpx/errors/throw_exception.hpp>
#include <hpx/modules/async_distributed.hpp>
#include <hpx/runtime_components/component_factory.hpp>
#include <hpx/runtime_components/new.hpp>
#include <hpx/runtime_distributed/server/runtime_support.hpp>
#include <hpx/type_support/extra_data.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace hpx::util {

    // This is explicitly instantiated to ensure that the id is stable across
    // shared libraries.
    extra_data_id_type
    extra_data_helper<collectives::detail::communicator_data>::id() noexcept
    {
        static std::uint8_t id = 0;
        return &id;
    }
}    // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
using collectives_component =
    hpx::components::component<hpx::collectives::detail::communicator_server>;

HPX_REGISTER_COMPONENT(collectives_component)

namespace hpx::collectives {

    namespace detail {

        communicator_server::communicator_server() noexcept    //-V730
          : num_sites_(0)
        {
            HPX_ASSERT(false);    // shouldn't ever be called
        }

        communicator_server::communicator_server(
            std::size_t num_sites, char const* basename) noexcept
          : gate_(num_sites)
          , num_sites_(num_sites)
          , basename_(basename)
        {
            HPX_ASSERT(
                num_sites != 0 && num_sites != static_cast<std::size_t>(-1));
        }

        communicator_server::~communicator_server() = default;
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    void communicator::set_info(
        num_sites_arg num_sites, this_site_arg this_site) noexcept
    {
        auto& [num_sites_, this_site_] =
            get_extra_data<detail::communicator_data>();

        num_sites_ = num_sites;
        this_site_ = this_site;
    }

    std::pair<num_sites_arg, this_site_arg> communicator::get_info()
        const noexcept
    {
        auto const* client_data =
            try_get_extra_data<detail::communicator_data>();

        if (client_data != nullptr)
        {
            return std::make_pair(
                client_data->num_sites_, client_data->this_site_);
        }

        return std::make_pair(num_sites_arg{}, this_site_arg{});
    }

    ///////////////////////////////////////////////////////////////////////////
    communicator create_communicator(char const* basename,
        num_sites_arg num_sites, this_site_arg this_site,
        generation_arg generation, root_site_arg root_site)
    {
        if (num_sites == static_cast<std::size_t>(-1))
        {
            num_sites = agas::get_num_localities(hpx::launch::sync);
        }
        if (this_site == static_cast<std::size_t>(-1))
        {
            this_site = agas::get_locality_id();
            if (root_site == static_cast<std::size_t>(-1))    //-V1051
            {
                root_site = this_site;
            }
        }

        HPX_ASSERT(this_site < num_sites);
        HPX_ASSERT(
            root_site != static_cast<std::size_t>(-1) && root_site < num_sites);

        std::string name(basename);
        if (generation != static_cast<std::size_t>(-1))
        {
            name += std::to_string(generation) + "/";
        }

        if (this_site == root_site)
        {
            // create a new communicator
            auto c = hpx::local_new<communicator>(num_sites, basename);

            // register the communicator's id using the given basename, this
            // keeps the communicator alive
            auto f = c.register_as(
                hpx::detail::name_from_basename(HPX_MOVE(name), this_site));

            return f.then(hpx::launch::sync,
                [=, target = HPX_MOVE(c)](hpx::future<bool>&& fut) mutable {
                    if (bool const result = fut.get(); !result)
                    {
                        HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                            "hpx::collectives::detail::create_communicator",
                            "the given base name for the communicator "
                            "operation was already registered: {}",
                            target.registered_name());
                    }
                    target.set_info(num_sites, this_site);
                    return target;
                });
        }

        // find existing communicator
        return hpx::find_from_basename<communicator>(HPX_MOVE(name), root_site);
    }

    ///////////////////////////////////////////////////////////////////////////
    communicator create_local_communicator(char const* basename,
        num_sites_arg num_sites, this_site_arg this_site,
        generation_arg generation, root_site_arg root_site)
    {
        if (root_site == static_cast<std::size_t>(-1))
        {
            root_site = this_site;
        }

        HPX_ASSERT(this_site < num_sites);
        HPX_ASSERT(
            root_site != static_cast<std::size_t>(-1) && root_site < num_sites);
        HPX_ASSERT(basename != nullptr && basename[0] != '\0');

        // make sure the communicator will be registered in the local AGAS
        // symbol service instance
        std::string name = hpx::util::format("/{}{}{}", agas::get_locality_id(),
            basename[0] == '/' ? "" : "/", basename);
        if (generation != static_cast<std::size_t>(-1))
        {
            name += std::to_string(generation) + "/";
        }

        if (this_site == root_site)
        {
            // create a new communicator
            auto c = hpx::local_new<communicator>(num_sites, basename);

            // register the communicator's id using the given basename, this
            // keeps the communicator alive
            bool const result = c.register_as(hpx::launch::sync,
                hpx::detail::name_from_basename(HPX_MOVE(name), this_site));

            if (!result)
            {
                HPX_THROW_EXCEPTION(hpx::error::bad_parameter,
                    "hpx::collectives::detail::create_local_communicator",
                    "the given base name for the communicator operation was "
                    "already registered: {}",
                    c.registered_name());
            }

            c.set_info(num_sites, this_site);
            return c;
        }

        // find existing communicator
        return hpx::find_from_basename<communicator>(HPX_MOVE(name), root_site);
    }
}    // namespace hpx::collectives

#endif    // !HPX_COMPUTE_DEVICE_CODE
