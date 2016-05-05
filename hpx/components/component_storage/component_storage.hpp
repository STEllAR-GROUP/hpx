//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENT_STORAGE_FEB_06_2015_0959AM)
#define HPX_COMPONENT_STORAGE_FEB_06_2015_0959AM

#include <hpx/include/components.hpp>
#include <hpx/include/naming.hpp>
#include <hpx/include/lcos.hpp>

#include <hpx/components/component_storage/server/component_storage.hpp>

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
        hpx::future<std::vector<char> > migrate_from_here(
            naming::gid_type const&);

        naming::id_type migrate_to_here_sync(std::vector<char> const&,
            naming::id_type const&, naming::address const&);
        std::vector<char> migrate_from_here_sync(naming::gid_type const&);

        future<std::size_t> size() const;
        std::size_t size_sync() const;
    };
}}

#endif


