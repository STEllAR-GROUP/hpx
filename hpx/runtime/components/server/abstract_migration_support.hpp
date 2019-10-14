//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_ABSTRACT_MIGRATION_SUPPORT_APR_04_2019_1203PM)
#define HPX_COMPONENTS_SERVER_ABSTRACT_MIGRATION_SUPPORT_APR_04_2019_1203PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/pinned_ptr.hpp>
#include <hpx/runtime/components/server/migration_support.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/traits/action_decorate_function.hpp>
#include <hpx/functional/bind_front.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// This hook has to be inserted into the derivation chain of any
    /// abstract_component_base for it to support migration.
    template <typename BaseComponent, typename Mutex = lcos::local::spinlock>
    struct abstract_base_migration_support : BaseComponent
    {
    private:
        typedef BaseComponent base_type;
        typedef typename base_type::this_component_type this_component_type;

    public:
        virtual ~abstract_base_migration_support() = default;

        // This component type supports migration.
//         HPX_CONSTEXPR static bool supports_migration() { return true; }

        // Pinning functionality
        virtual void pin() = 0;
        virtual bool unpin() = 0;
        virtual std::uint32_t pin_count() const = 0;
        virtual void mark_as_migrated() = 0;

        // migration support
        virtual hpx::future<void> mark_as_migrated(
            hpx::id_type const& to_migrate) = 0;
        virtual void on_migrated() = 0;

        typedef void decorates_action;

        /// This is the hook implementation for decorate_action which makes
        /// sure that the object becomes pinned during the execution of an
        /// action.
        template <typename F>
        static threads::thread_function_type
        decorate_action(naming::address::address_type lva, F && f)
        {
            // Make sure we pin the component at construction of the bound object
            // which will also unpin it once the thread runs to completion (the
            // bound object goes out of scope).
            return util::one_shot(util::bind_front(
                &abstract_base_migration_support::thread_function,
                get_lva<this_component_type>::call(lva),
                traits::component_decorate_function<base_type>::call(
                    lva, std::forward<F>(f)),
                components::pinned_ptr::create<this_component_type>(lva)));
        }

    protected:
        // Execute the wrapped action. This function is bound in decorate_action
        // above. The bound object performs the pinning/unpinning.
        threads::thread_result_type thread_function(
            threads::thread_function_type && f,
            components::pinned_ptr,
            threads::thread_state_ex_enum state)
        {
            return f(state);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    /// This hook has to be inserted into the derivation chain of any component
    /// for it to support migration.
    template <typename Derived, typename Base>
    struct abstract_migration_support : migration_support<Derived>, Base
    {
        using base_type = migration_support<Derived>;
        using abstract_base_type = Base;

        using wrapping_type = typename base_type::wrapping_type;
        using wrapped_type = typename base_type::wrapped_type;

        using type_holder = Derived;
        using base_type_holder = Base;

        template <typename ... Ts>
        abstract_migration_support(Ts&&... ts)
          : abstract_base_type(std::forward<Ts>(ts)...)
        {
        }

        ~abstract_migration_support() = default;

        HPX_CXX14_CONSTEXPR void finalize() {}

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
}}

#endif
