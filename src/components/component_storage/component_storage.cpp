//  Copyright (c) 2015-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/include/components.hpp>

#include <hpx/components/component_storage/component_storage.hpp>

namespace hpx { namespace components
{
    component_storage::component_storage(hpx::id_type target_locality)
      : base_type(hpx::new_<server::component_storage>(target_locality))
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

    ///////////////////////////////////////////////////////////////////////////
    // store and load from disk
    future<void> component_storage::write_to_disk(std::string const& filename) const
    {
        typedef server::component_storage::write_to_disk_action action_type;
        return hpx::async<action_type>(this->get_id(), filename);
    }

    void component_storage::write_to_disk_sync(std::string const& filename) const
    {
        write_to_disk(filename).get();
    }

    future<void> component_storage::read_from_disk(std::string const& filename)
    {
        typedef server::component_storage::read_from_disk_action action_type;
        return hpx::async<action_type>(this->get_id(), filename);
    }

    void component_storage::read_from_disk_sync(std::string const& filename)
    {
        read_from_disk(filename).get();
    }
}}
