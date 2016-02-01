//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENT_STORAGE_SERVER_FEB_04_2015_0143PM)
#define HPX_COMPONENT_STORAGE_SERVER_FEB_04_2015_0143PM

#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/unordered_map.hpp>

#include <hpx/components/component_storage/export_definitions.hpp>

#include <vector>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_MIGRATE_TO_STORAGE_EXPORT component_storage
      : public simple_component_base<component_storage>
    {
        typedef lcos::local::spinlock mutex_type;
        typedef
            hpx::unordered_map<
                naming::gid_type,
                std::pair<components::component_type, std::vector<char> >
            >
            data_store_type;

    public:
        component_storage();
        component_storage(hpx::id_type const& locality);
        component_storage(std::vector<hpx::id_type> const& localities);

        ~component_storage();

        naming::gid_type migrate_to_here(std::vector<char> const&,
            naming::id_type const& , naming::address const&);
        std::vector<char> migrate_from_here(naming::gid_type const& );
        std::size_t size() const { return data_.size(); }

        void write_to_disk(std::string const& filename) const;
        void read_from_disk(std::string const& filename);

        HPX_DEFINE_COMPONENT_ACTION(component_storage, migrate_to_here);
        HPX_DEFINE_COMPONENT_ACTION(component_storage, migrate_from_here);
        HPX_DEFINE_COMPONENT_ACTION(component_storage, size);

        HPX_DEFINE_COMPONENT_ACTION(component_storage, write_to_disk);
        HPX_DEFINE_COMPONENT_ACTION(component_storage, read_from_disk);

    private:
        data_store_type data_;
    };
}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::component_storage::migrate_to_here_action,
    component_storage_migrate_component_to_here_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::component_storage::migrate_from_here_action,
    component_storage_migrate_component_from_here_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::component_storage::size_action,
    component_storage_size_action);

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::component_storage::write_to_disk_action,
    component_storage_write_to_disk_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::component_storage::read_from_disk_action,
    component_storage_read_from_disk_action);

typedef std::pair<hpx::components::component_type, std::vector<char> >
    hpx_component_storage_data_type;
HPX_REGISTER_UNORDERED_MAP_DECLARATION(
    hpx::naming::gid_type, hpx_component_storage_data_type)

#endif


