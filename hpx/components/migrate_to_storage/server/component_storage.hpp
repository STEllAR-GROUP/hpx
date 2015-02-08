//  Copyright (c) 2007-2015 Hartmut Kaiser
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

#include <hpx/components/migrate_to_storage/export_definitions.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace std
{
    // specialize std::hash for hpx::naming::gid_type
    template <>
    struct hash<hpx::naming::gid_type>
    {
        typedef hpx::naming::gid_type argument_type;
        typedef std::size_t result_type;

        result_type operator()(argument_type const& gid) const
        {
            result_type const h1 (std::hash<boost::uint64_t>()(gid.get_lsb()));
            result_type const h2 (std::hash<boost::uint64_t>()(gid.get_msb()));
            return h1 ^ (h2 << 1);
        }
    };
}

namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_MIGRATE_TO_STORAGE_EXPORT component_storage
      : public simple_component_base<component_storage>
    {
        typedef lcos::local::spinlock mutex_type;

    public:
        component_storage() {}

        naming::gid_type migrate_to_here(std::vector<char> const&,
            naming::id_type, naming::address const&);
        std::vector<char> migrate_from_here(naming::gid_type);

        HPX_DEFINE_COMPONENT_ACTION(component_storage, migrate_to_here);
        HPX_DEFINE_COMPONENT_ACTION(component_storage, migrate_from_here);

    private:
        hpx::unordered_map<naming::gid_type, std::vector<char> > data_;
    };
}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::component_storage::migrate_to_here_action,
    migrate_component_to_here_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::component_storage::migrate_from_here_action,
    migrate_component_from_here_action);

typedef std::vector<char> hpx_component_storage_data_type;
HPX_REGISTER_UNORDERED_MAP_DECLARATION(
    hpx::naming::gid_type, hpx_component_storage_data_type)

#endif


