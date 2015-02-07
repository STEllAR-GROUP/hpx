//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/components/migrate_to_storage/server/component_storage.hpp>

namespace hpx { namespace components { namespace server
{
    naming::gid_type component_storage::migrate_to_here(
        std::vector<char> const& data, naming::id_type id)
    {
        mutex_type::scoped_lock l(mtx_);
        data_[id.get_gid()] = data;

        id.make_unmanaged();            // we can now release the object

        return naming::invalid_gid;
    }

    std::vector<char> component_storage::migrate_from_here(naming::gid_type id)
    {
        mutex_type::scoped_lock l(mtx_);
        std::vector<char> data(std::move(data_[id]));
        data_.erase(id);
        return data;
    }
}}}

HPX_REGISTER_UNORDERED_MAP(hpx::naming::gid_type, hpx_component_storage_data_type)
