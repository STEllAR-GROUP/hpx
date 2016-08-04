////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/runtime/agas/detail/bootstrap_component_namespace.hpp>

#include <string>
#include <vector>

namespace hpx { namespace agas { namespace detail
{
    naming::address bootstrap_component_namespace::addr() const
    {
        return naming::address(
            hpx::get_locality(),
            server::component_namespace::get_component_type(),
            this->ptr()
        );
    }

    naming::id_type bootstrap_component_namespace::gid() const
    {
        return naming::id_type(
            naming::gid_type(HPX_AGAS_COMPONENT_NS_MSB, HPX_AGAS_COMPONENT_NS_LSB),
            naming::id_type::unmanaged);
    }

    components::component_type bootstrap_component_namespace::bind_prefix(
        std::string const& key, boost::uint32_t prefix)
    {
        return server_.bind_prefix(key, prefix);
    }

    components::component_type
    bootstrap_component_namespace::bind_name(std::string const& name)
    {
        return server_.bind_name(name);
    }

    std::vector<boost::uint32_t>
    bootstrap_component_namespace::resolve_id(components::component_type key)
    {
        return server_.resolve_id(key);
    }

    bool bootstrap_component_namespace::unbind(std::string const& key)
    {
        return server_.unbind(key);
    }

    void
    bootstrap_component_namespace::iterate_types(
        iterate_types_function_type const& f)
    {
        return server_.iterate_types(f);
    }

    std::string
    bootstrap_component_namespace::get_component_type_name(
        components::component_type type)
    {
        return server_.get_component_type_name(type);
    }

    lcos::future<boost::uint32_t> bootstrap_component_namespace::get_num_localities(
        components::component_type type)
    {
        return hpx::make_ready_future(server_.get_num_localities(type));
    }

    naming::gid_type bootstrap_component_namespace::statistics_counter(
        std::string const& name)
    {
        return server_.statistics_counter(name);
    }

    void bootstrap_component_namespace::register_counter_types()
    {
        server::component_namespace::register_counter_types();
        server::component_namespace::register_global_counter_types();
    }

    void
    bootstrap_component_namespace::register_server_instance(boost::uint32_t locality_id)
    {
        HPX_ASSERT(locality_id == 0);
        const char* servicename("locality#0/");
        server_.register_server_instance(servicename);
    }

    void bootstrap_component_namespace::unregister_server_instance(error_code& ec)
    {
        server_.unregister_server_instance(ec);
    }
}}}
