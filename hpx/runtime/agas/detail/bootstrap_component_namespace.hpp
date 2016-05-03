////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_RUNTIME_AGAS_BOOTSTRAP_COMPONENT_NAMESPACE_HPP)
#define HPX_RUNTIME_AGAS_BOOTSTRAP_COMPONENT_NAMESPACE_HPP

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/agas/component_namespace.hpp>
#include <hpx/runtime/agas/server/component_namespace.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/function.hpp>

#include <boost/cstdint.hpp>

#include <string>
#include <vector>

namespace hpx { namespace agas { namespace detail
{
    struct bootstrap_component_namespace
        : component_namespace
    {
        typedef hpx::util::function<
            void(std::string const&, components::component_type)
        > iterate_types_function_type;

        naming::address::address_type ptr() const
        {
            return reinterpret_cast<naming::address::address_type>(&server_);
        }
        naming::address addr() const;
        naming::id_type gid() const;

        components::component_type bind_prefix(
            std::string const& key, boost::uint32_t prefix);

        components::component_type bind_name(std::string const& name);

        std::vector<boost::uint32_t> resolve_id(components::component_type key);

        bool unbind(std::string const& key);

        void iterate_types(iterate_types_function_type const& f);

        std::string get_component_type_name(components::component_type type);

        lcos::future<boost::uint32_t> get_num_localities(
            components::component_type type);

        naming::gid_type statistics_counter(std::string const& name);

        void register_counter_types();

        void register_server_instance(boost::uint32_t locality_id);

        void unregister_server_instance(error_code& ec);

    private:
        server::component_namespace server_;
    };

}}}

#endif

