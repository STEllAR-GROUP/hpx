//  Copyright (c) 2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include <hpx/components/component_storage/server/component_storage.hpp>

#include <cstddef>
#include <vector>

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_MIGRATE_TO_STORAGE_EXPORT component_storage
      : public client_base<component_storage, server::component_storage>
    {
        typedef client_base<component_storage, server::component_storage>
            base_type;

    public:
        component_storage(hpx::id_type target_locality);
        component_storage(hpx::future<naming::id_type> && f);

        hpx::future<naming::id_type> migrate_to_here(std::vector<char> const&,
            naming::id_type const&, naming::address const&);
        naming::id_type migrate_to_here(launch::sync_policy,
            std::vector<char> const&, naming::id_type const&,
            naming::address const&);

        hpx::future<std::vector<char> > migrate_from_here(
            naming::gid_type const&);
        std::vector<char> migrate_from_here(launch::sync_policy,
            naming::gid_type const&);

        future<std::size_t> size() const;
        std::size_t size(launch::sync_policy) const;
    };
}}



