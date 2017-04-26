//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_RUNTIME_COMPONENTS_FWD_HPP
#define HPX_RUNTIME_COMPONENTS_FWD_HPP

#include <hpx/config.hpp>
#include <hpx/traits/managed_component_policies.hpp>

#include <cstddef>
#include <string>

namespace hpx
{
    enum logging_destination
    {
        destination_hpx = 0,
        destination_timing = 1,
        destination_agas = 2,
        destination_parcel = 3,
        destination_app = 4,
        destination_debuglog = 5
    };

    /// \namespace components
    namespace components
    {
        /// \ cond NODETAIL
        namespace detail
        {
            struct this_type {};
        }
        /// \ endcond

        ///////////////////////////////////////////////////////////////////////
        template <typename Component = detail::this_type>
        class fixed_component_base;

        template <typename Component>
        class fixed_component;

        template <typename Component = detail::this_type>
        class abstract_simple_component_base;

        template <typename Component = detail::this_type>
        class simple_component_base;

        template <typename Component>
        class simple_component;

        template <typename Component, typename Derived = detail::this_type>
        class abstract_managed_component_base;

        template <typename Component, typename Wrapper = detail::this_type,
            typename CtorPolicy = traits::construct_without_back_ptr,
            typename DtorPolicy = traits::managed_object_controls_lifetime>
        class managed_component_base;

        template <typename Component, typename Derived = detail::this_type>
        class managed_component;

        struct HPX_API_EXPORT component_factory_base;

        template <typename Component>
        struct component_factory;

        class runtime_support;
        class memory;
        class memory_block;

        class pinned_ptr;

        namespace stubs
        {
            struct runtime_support;
            struct memory;
            struct memory_block;
        }

        namespace server
        {
            class HPX_API_EXPORT runtime_support;
            class HPX_API_EXPORT memory;
            class HPX_API_EXPORT memory_block;
        }

        HPX_EXPORT void console_logging(logging_destination dest,
            std::size_t level, std::string const& msg);
        HPX_EXPORT void cleanup_logging();
        HPX_EXPORT void activate_logging();
    }

    HPX_EXPORT components::server::runtime_support* get_runtime_support_ptr();
}

#endif
