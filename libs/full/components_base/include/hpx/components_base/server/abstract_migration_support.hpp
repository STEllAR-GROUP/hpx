//  Copyright (c) 2019-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/components_base/pinned_ptr.hpp>
#include <hpx/components_base/server/migration_support.hpp>
#include <hpx/components_base/traits/action_decorate_function.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/threading_base.hpp>
#include <hpx/naming_base/id_type.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace hpx { namespace components {

    ///////////////////////////////////////////////////////////////////////////
    /// This hook has to be inserted into the derivation chain of any
    /// abstract_component_base for it to support migration.
    template <typename BaseComponent, typename Mutex = hpx::spinlock>
    struct abstract_base_migration_support : BaseComponent
    {
    private:
        using base_type = BaseComponent;
        using this_component_type = typename base_type::this_component_type;

    public:
        virtual ~abstract_base_migration_support() = default;

        // This component type supports migration.
        // static constexpr bool supports_migration() { return true; }

        // Pinning functionality
        virtual void pin() = 0;
        virtual bool unpin() = 0;
        virtual std::uint32_t pin_count() const = 0;
        virtual void mark_as_migrated() = 0;

        // migration support
        virtual hpx::future<void> mark_as_migrated(
            hpx::id_type const& to_migrate) = 0;
        virtual void on_migrated() = 0;

        using decorates_action = void;

        // This is the hook implementation for decorate_action which makes
        // sure that the object becomes pinned during the execution of an
        // action.
        template <typename F>
        static threads::thread_function_type decorate_action(
            naming::address_type lva, F&& f)
        {
            // Make sure we pin the component at construction of the bound object
            // which will also unpin it once the thread runs to completion (the
            // bound object goes out of scope).
            return util::one_shot(hpx::bind_front(
                &abstract_base_migration_support::thread_function,
                get_lva<this_component_type>::call(lva),
                traits::component_decorate_function<base_type>::call(
                    lva, HPX_FORWARD(F, f)),
                components::pinned_ptr::create<this_component_type>(lva)));
        }

    protected:
        // Execute the wrapped action. This function is bound in decorate_action
        // above. The bound object performs the pinning/unpinning.
        threads::thread_result_type thread_function(
            threads::thread_function_type&& f, components::pinned_ptr,
            threads::thread_restart_state state)
        {
            return f(state);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    /// This hook has to be inserted into the derivation chain of any component
    /// for it to support migration.
    template <typename Derived, typename Base>
    struct abstract_migration_support
      : migration_support<Derived>
      , Base
    {
        using base_type = migration_support<Derived>;
        using abstract_base_type = Base;

        using wrapping_type = typename base_type::wrapping_type;
        using wrapped_type = typename base_type::wrapped_type;

        using type_holder = Derived;
        using base_type_holder = Base;

        using base_type::get_current_address;

        abstract_migration_support() = default;

        template <typename T, typename... Ts,
            typename Enable = std::enable_if_t<
                !std::is_same_v<std::decay_t<T>, abstract_migration_support>>>
        abstract_migration_support(T&& t, Ts&&... ts)
          : abstract_base_type(HPX_FORWARD(T, t), HPX_FORWARD(Ts, ts)...)
        {
        }

        ~abstract_migration_support() = default;

        constexpr void finalize() noexcept {}

        hpx::future<void> mark_as_migrated(
            hpx::id_type const& to_migrate) override
        {
            return this->base_type::mark_as_migrated(to_migrate);
        }

        void mark_as_migrated() override
        {
            this->base_type::mark_as_migrated();
        }

        std::uint32_t pin_count() const override
        {
            return this->base_type::pin_count();
        }

        void pin() override
        {
            this->base_type::pin();
        }
        bool unpin() override
        {
            return this->base_type::unpin();
        }

        void on_migrated() override
        {
            return this->base_type::on_migrated();
        }
    };
}}    // namespace hpx::components
