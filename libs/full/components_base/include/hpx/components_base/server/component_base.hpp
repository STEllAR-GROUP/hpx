//  Copyright (c) 2015 Thomas Heller
//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/components_base/agas_interface.hpp>
#include <hpx/components_base/component_type.hpp>
#include <hpx/components_base/components_base_fwd.hpp>
#include <hpx/components_base/traits/is_component.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/naming_base/address.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace components {

    namespace detail {

        struct base_component : traits::detail::component_tag
        {
            constexpr base_component() = default;
            HPX_EXPORT ~base_component();

            base_component(base_component const&) = default;
            base_component& operator=(base_component const&) = default;

            // just move our gid_
            base_component(base_component&& rhs) noexcept = default;
            base_component& operator=(base_component&& rhs) noexcept = default;

            // finalize() will be called just before the instance gets destructed
            static constexpr void finalize() noexcept {}

            HPX_EXPORT hpx::id_type get_id(naming::gid_type gid) const;
            HPX_EXPORT hpx::id_type get_unmanaged_id(
                naming::gid_type const& gid) const;

            static void mark_as_migrated() noexcept
            {
                // If this assertion is triggered then this component instance
                // is being migrated even if the component type has not been
                // enabled to support migration.
                HPX_ASSERT(false);
            }

            static void on_migrated() noexcept
            {
                // If this assertion is triggered then this component instance
                // is being migrated even if the component type has not been
                // enabled to support migration.
                HPX_ASSERT(false);
            }

        protected:
            // Create a new GID (if called for the first time), assign this
            // GID to this instance of a component and register this gid
            // with the AGAS service
            //
            // Returns he global id (GID) assigned to this instance of a component
            HPX_EXPORT naming::gid_type get_base_gid_dynamic(
                naming::gid_type const& assign_gid, naming::address const& addr,
                naming::gid_type (*f)(naming::gid_type) = nullptr) const;

            HPX_EXPORT naming::gid_type get_base_gid(
                naming::address const& addr,
                naming::gid_type (*f)(naming::gid_type) = nullptr) const;

        protected:
            mutable naming::gid_type gid_;
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    class component_base : public detail::base_component
    {
    protected:
        using this_component_type = std::conditional_t<
            std::is_same_v<Component, components::detail::this_type>,
            component_base, Component>;

    public:
        using wrapped_type = this_component_type;
        using base_type_holder = this_component_type;
        using wrapping_type = component<this_component_type>;

        // Construct an empty component
        constexpr component_base() = default;

        // Destruct a component
        ~component_base() = default;

        component_base(component_base const&) = default;
        component_base(component_base&& rhs) noexcept = default;

        component_base& operator=(component_base const&) = default;
        component_base& operator=(component_base&& rhs) noexcept = default;

    public:
        naming::address get_current_address() const
        {
            return naming::address(
                naming::get_gid_from_locality_id(agas::get_locality_id()),
                components::get_component_type<wrapped_type>(),
                const_cast<Component*>(static_cast<Component const*>(this)));
        }

        hpx::id_type get_id() const
        {
            // all credits should have been taken already
            return this->detail::base_component::get_id(
                static_cast<Component const&>(*this).get_base_gid());
        }

        hpx::id_type get_unmanaged_id() const
        {
            return this->detail::base_component::get_unmanaged_id(
                static_cast<Component const&>(*this).get_base_gid());
        }

        naming::gid_type get_base_gid(
            naming::gid_type const& assign_gid = naming::invalid_gid) const
        {
            HPX_ASSERT(!assign_gid);    // migration is not supported here
            HPX_UNUSED(assign_gid);
            return this->detail::base_component::get_base_gid(
                static_cast<Component const&>(*this).get_current_address());
        }
    };
}}    // namespace hpx::components
