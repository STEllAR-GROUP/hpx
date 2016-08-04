////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/async.hpp>
#include <hpx/runtime/agas/detail/hosted_component_namespace.hpp>
#include <hpx/runtime/agas/server/component_namespace.hpp>
#include <hpx/runtime/serialization/vector.hpp>

#include <string>
#include <vector>

namespace hpx { namespace agas { namespace detail
{
    hosted_component_namespace::hosted_component_namespace(naming::address addr)
      : gid_(naming::gid_type(HPX_AGAS_COMPONENT_NS_MSB, HPX_AGAS_COMPONENT_NS_LSB),
            naming::id_type::unmanaged)
      , addr_(addr)
    {
    }

    components::component_type hosted_component_namespace::bind_prefix(
        std::string const& key, boost::uint32_t prefix)
    {
        server::component_namespace::bind_prefix_action action;
        return action(gid_, key, prefix);
    }

    components::component_type
    hosted_component_namespace::bind_name(std::string const& name)
    {
        server::component_namespace::bind_name_action action;
        return action(gid_, name);
    }

    std::vector<boost::uint32_t>
    hosted_component_namespace::resolve_id(components::component_type key)
    {
        server::component_namespace::resolve_id_action action;
        return action(gid_, key);
    }

    bool hosted_component_namespace::unbind(std::string const& key)
    {
        server::component_namespace::unbind_action action;
        return action(gid_, key);
    }

    void
    hosted_component_namespace::iterate_types(iterate_types_function_type const& f)
    {
        server::component_namespace::iterate_types_action action;
        return action(gid_, f);
    }

    std::string
    hosted_component_namespace::get_component_type_name(
        components::component_type type)
    {
        server::component_namespace::get_component_type_name_action action;
        return action(gid_, type);
    }

    lcos::future<boost::uint32_t> hosted_component_namespace::get_num_localities(
        components::component_type type)
    {
        server::component_namespace::get_num_localities_action action;
        return hpx::async(action, gid_, type);
    }

    naming::gid_type hosted_component_namespace::statistics_counter(
        std::string const& name)
    {
        server::component_namespace::statistics_counter_action action;
        return action(gid_, name).get_gid();
    }
}}}
