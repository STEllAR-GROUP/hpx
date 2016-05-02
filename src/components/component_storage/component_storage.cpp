//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/include/components.hpp>

#include <hpx/components/component_storage/component_storage.hpp>

#include <vector>

namespace hpx { namespace components
{
    component_storage::component_storage(hpx::id_type target_locality)
      : base_type(hpx::new_<server::component_storage>(target_locality))
    {}

    component_storage::component_storage(hpx::future<naming::id_type> && f)
      : base_type(std::move(f))
    {}

    ///////////////////////////////////////////////////////////////////////////
    hpx::future<naming::id_type> component_storage::migrate_to_here(
        std::vector<char> const& data, naming::id_type id,
        naming::address const& addr)
    {
        typedef server::component_storage::migrate_to_here_action action_type;
        return hpx::async<action_type>(this->get_id(), data, id, addr);
    }

    naming::id_type component_storage::migrate_to_here_sync(
        std::vector<char> const& data, naming::id_type id,
        naming::address const& addr)
    {
        return migrate_to_here(data, id, addr).get();
    }

    hpx::future<std::vector<char> > component_storage::migrate_from_here(
        naming::gid_type id)
    {
        typedef server::component_storage::migrate_from_here_action action_type;
        return hpx::async<action_type>(this->get_id(), id);
    }

    std::vector<char> component_storage::migrate_from_here_sync(
        naming::gid_type id)
    {
        return migrate_from_here(id).get();
    }

    hpx::future<std::size_t> component_storage::size() const
    {
        typedef server::component_storage::size_action action_type;
        return hpx::async<action_type>(this->get_id());
    }

    std::size_t component_storage::size_sync() const
    {
        return size().get();
    }
}}
