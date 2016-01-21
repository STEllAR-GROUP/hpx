//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_MIGRATION_SUPPORT_FEB_03_2014_0230PM)
#define HPX_COMPONENTS_SERVER_MIGRATION_SUPPORT_FEB_03_2014_0230PM

#include <hpx/config.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/threads_fwd.hpp>
#include <hpx/util/bind.hpp>

#include <boost/thread/locks.hpp>

#include <type_traits>

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
        {}

        ~migration_support()
        {
            // prevent base destructor from unregistering the gid if this
            // instance has been migrated
            if (pin_count_ == ~0x0u)
                this->gid_ = naming::invalid_gid;
        }

        // This component type supports migration.
        static HPX_CONSTEXPR bool supports_migration() { return true; }

        // Pinning functionality
        void pin()
        {
            boost::lock_guard<mutex_type> l(mtx_);
            ++pin_count_;
        }
        void unpin()
        {
            boost::lock_guard<mutex_type> l(mtx_);
            HPX_ASSERT(pin_count_ != 0);
            if (pin_count_ != ~0x0u)
            {
                if (--pin_count_ == 0)
                {
                    // trigger pending migration if this was the last unpin and
                    // a migration operation is pending
                    if (trigger_migration_.valid() && migration_target_locality_)
                    {
                        migration_target_locality_ = naming::invalid_id;
                        agas::mark_as_migrated(this->base_type::get_unmanaged_id());
                        trigger_migration_.set_value();
                    }
                }
            }
        }

        boost::uint32_t pin_count() const
        {
            boost::lock_guard<mutex_type> l(mtx_);
            return pin_count_;
        }
        void mark_as_migrated()
        {
            boost::lock_guard<mutex_type> l(mtx_);
            HPX_ASSERT(1 == pin_count_);
            pin_count_ = ~0x0u;
        }

        hpx::future<void>
        mark_as_migrated(hpx::id_type const& target_location)
        {
            boost::lock_guard<mutex_type> l(mtx_);

            // make sure that no migration is currently in flight
            if (migration_target_locality_)
            {
                return make_exceptional_future<void>(
                    HPX_GET_EXCEPTION(invalid_status,
                        "migration_support::mark_as_migrated",
                        "migration operation is already in flight"));
            }

            if (1 == pin_count_)
            {
                // all is well, migration can be triggered now
                agas::mark_as_migrated(this->base_type::get_unmanaged_id());
                return make_ready_future();
            }

            // delay migrate operation until pin count goes to zero
            migration_target_locality_ = target_location;
            return trigger_migration_.get_future();
        }

        /// This is the hook implementation for decorate_action which makes
        /// sure that the object becomes pinned during the execution of an
        /// action.
        template <typename F>
        static threads::thread_function_type
        decorate_action(naming::address::address_type lva, F && f)
        {
            this_component_type* this_ =
                get_lva<this_component_type>::call(lva);

            // pin the component whenever a thread is being scheduled
            this_->pin();

            // make sure we unpin the component once the thread runs to
            // completion
            return util::bind(
                util::one_shot(&migration_support::thread_function),
                this_, util::placeholders::_1,
                base_type::decorate_action(lva, std::forward<F>(f)));
        }

    protected:
        struct scoped_unpinner
        {
            scoped_unpinner(migration_support& outer)
              : outer_(outer)
            {}

            ~scoped_unpinner()
            {
                // unpin component once thread runs to completion, this may
                // trigger any pending migration operation
                outer_.unpin();
            }

            migration_support& outer_;
        };

        // Execute the wrapped action. This pins the object making sure that
        // no migration will happen while an operation is in flight.
        threads::thread_state_enum thread_function(
            threads::thread_state_ex_enum state,
            threads::thread_function_type && f)
        {
            scoped_unpinner sp(*this);
            return f(state);
        }

    private:
        mutable mutex_type mtx_;
        boost::uint32_t pin_count_;
        hpx::lcos::local::promise<void> trigger_migration_;
        hpx::id_type migration_target_locality_;
    };
}}

#endif
