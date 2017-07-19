//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_MIGRATION_SUPPORT_FEB_03_2014_0230PM)
#define HPX_COMPONENTS_SERVER_MIGRATION_SUPPORT_FEB_03_2014_0230PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/promise.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/components/pinned_ptr.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/traits/action_decorate_function.hpp>
#include <hpx/util/bind.hpp>

#include <cstdint>
#include <mutex>
#include <type_traits>
#include <utility>

namespace hpx { namespace components
{
    /// This hook has to be inserted into the derivation chain of any component
    /// for it to support migration.
    template <typename BaseComponent, typename Mutex = lcos::local::spinlock>
    struct migration_support : BaseComponent
    {
    private:
        typedef Mutex mutex_type;
        typedef BaseComponent base_type;
        typedef typename base_type::this_component_type this_component_type;

    public:
        template <typename ...Arg>
        migration_support(Arg &&... arg)
          : base_type(std::forward<Arg>(arg)...)
          , pin_count_(0)
          , was_marked_for_migration_(false)
        {}

        ~migration_support()
        {
            // prevent base destructor from unregistering the gid if this
            // instance has been migrated
            if (pin_count_ == ~0x0u)
                this->gid_ = naming::invalid_gid;
        }

        naming::gid_type get_base_gid(
            naming::gid_type const& assign_gid = naming::invalid_gid) const
        {
            // we don't store migrating objects in the AGAS cache
            naming::gid_type result = this->base_type::get_base_gid(assign_gid);
            naming::detail::set_dont_store_in_cache(result);
            return result;
        }

        // This component type supports migration.
        static HPX_CONSTEXPR bool supports_migration() { return true; }

        // Pinning functionality
        void pin()
        {
            std::lock_guard<mutex_type> l(mtx_);
            HPX_ASSERT(pin_count_ != ~0x0u);
            if (pin_count_ != ~0x0u)
                ++pin_count_;
        }
        void unpin()
        {
            // make sure to always grab the AGAS lock first
            agas::mark_as_migrated(this->gid_,
                [this]() mutable -> std::pair<bool, hpx::future<void> >
                {
                    std::unique_lock<mutex_type> l(mtx_);
                    HPX_ASSERT(this->pin_count_ != 0);
                    if (this->pin_count_ != ~0x0u)
                    {
                        if (--this->pin_count_ == 0)
                        {
                            // trigger pending migration if this was the last
                            // unpin and a migration operation is pending
                            if (trigger_migration_.valid() && was_marked_for_migration_)
                            {
                                was_marked_for_migration_ = false;

                                hpx::lcos::local::promise<void> p;
                                std::swap(p, trigger_migration_);

                                l.unlock();

                                p.set_value();
                                return std::make_pair(true, make_ready_future());
                            }
                        }
                    }
                    return std::make_pair(false, make_ready_future());
                });
        }

        std::uint32_t pin_count() const
        {
            std::lock_guard<mutex_type> l(mtx_);
            return pin_count_;
        }
        void mark_as_migrated()
        {
            std::lock_guard<mutex_type> l(mtx_);
            HPX_ASSERT(1 == pin_count_);
            pin_count_ = ~0x0u;
        }

        hpx::future<void> mark_as_migrated(hpx::id_type const& to_migrate)
        {
            // we need to first lock the AGAS migrated objects table, only then
            // access (lock) the object
            return agas::mark_as_migrated(to_migrate.get_gid(),
                [this]() mutable -> std::pair<bool, hpx::future<void> >
                {
                    std::unique_lock<mutex_type> l(mtx_);

                    // make sure that no migration is currently in flight
                    if (was_marked_for_migration_)
                    {
                        l.unlock();
                        return std::make_pair(false,
                            hpx::make_exceptional_future<void>(
                                HPX_GET_EXCEPTION(invalid_status,
                                    "migration_support::mark_as_migrated",
                                    "migration operation is already in flight")
                            ));
                    }

                    if (1 == pin_count_)
                    {
                        // all is well, migration can be triggered now
                        return std::make_pair(true, make_ready_future());
                    }

                    // delay migrate operation until pin count goes to zero
                    was_marked_for_migration_ = true;
                    hpx::future<void> f = trigger_migration_.get_future();

                    l.unlock();
                    return std::make_pair(true, std::move(f));
                });
        }

        /// This hook is invoked on the newly created object after the migration
        /// has been finished
        void on_migrated() {}

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
            return util::bind(
                util::one_shot(&migration_support::thread_function),
                get_lva<this_component_type>::call(lva),
                util::placeholders::_1,
                traits::action_decorate_function<base_type>::call(
                    lva, std::forward<F>(f)),
                components::pinned_ptr::create<this_component_type>(lva));
        }

        // Return whether the given object was migrated, if it was not
        // migrated, it also returns a pinned pointer.
        static std::pair<bool, components::pinned_ptr>
        was_object_migrated(hpx::naming::gid_type const& id,
            naming::address::address_type lva)
        {
            return agas::was_object_migrated(id,
                [lva]() -> components::pinned_ptr
                {
                    return components::pinned_ptr::create<this_component_type>(lva);
                });
        }

    protected:
        // Execute the wrapped action. This function is bound in decorate_action
        // above. The bound object performs the pinning/unpinning.
        threads::thread_result_type thread_function(
            threads::thread_state_ex_enum state,
            threads::thread_function_type && f,
            components::pinned_ptr)
        {
            return f(state);
        }

    private:
        mutable mutex_type mtx_;
        std::uint32_t pin_count_;
        hpx::lcos::local::promise<void> trigger_migration_;
        bool was_marked_for_migration_;
    };
}}

#endif
