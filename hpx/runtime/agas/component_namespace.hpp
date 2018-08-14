////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_RUNTIME_AGAS_COMPONENT_NAMESPACE_HPP)
#define HPX_RUNTIME_AGAS_COMPONENT_NAMESPACE_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/agas_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/naming/name.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace hpx { namespace agas
{
    struct component_namespace
    {
        virtual ~component_namespace();

        virtual naming::address::address_type ptr() const=0;
        virtual naming::address addr() const=0;
        virtual naming::id_type gid() const=0;

        virtual components::component_type bind_prefix(
            std::string const& key, std::uint32_t prefix)=0;

        virtual components::component_type bind_name(std::string const& name)=0;

        virtual std::vector<std::uint32_t> resolve_id(components::component_type key)=0;

        virtual bool unbind(std::string const& key)=0;

        virtual void iterate_types(iterate_types_function_type const& f)=0;

        virtual std::string get_component_type_name(components::component_type type)=0;

        virtual lcos::future<std::uint32_t>
        get_num_localities(components::component_type type)=0;

        virtual naming::gid_type statistics_counter(std::string const& name)=0;

        virtual void register_counter_types()
        {}

        virtual void register_server_instance(std::uint32_t locality_id)
        {}

        virtual void unregister_server_instance(error_code& ec)
        {}
    };

}}

#endif

