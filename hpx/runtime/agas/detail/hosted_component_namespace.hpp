
////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_RUNTIME_AGAS_HOSTED_COMPONENT_NAMESPACE_HPP)
#define HPX_RUNTIME_AGAS_HOSTED_COMPONENT_NAMESPACE_HPP

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/agas/component_namespace.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/util/function.hpp>

#include <boost/cstdint.hpp>

#include <string>
#include <vector>

namespace hpx { namespace agas { namespace detail
{
    struct hosted_component_namespace
        : component_namespace
    {
        explicit hosted_component_namespace(naming::address addr);
        hosted_component_namespace();

        naming::address::address_type ptr() const
        {
            return addr_.address_;
        }
        naming::address addr() const
        {
            return addr_;
        }
        naming::id_type gid() const
        {
            return gid_;
        }

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
    private:
        naming::id_type gid_;
        naming::address addr_;
    };

}}}

#endif
