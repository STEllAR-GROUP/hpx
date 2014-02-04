//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERVER_MIGRATABLE_FEB_03_2014_0230PM)
#define HPX_COMPONENTS_SERVER_MIGRATABLE_FEB_03_2014_0230PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/local/spinlock.hpp>

namespace hpx { namespace components
{
    /// This hook has to be inserted into the derivation chain of any component
    /// for it to support migration.
    template <typename BaseComponent, typename Mutex = lcos::local::spinlock>
    struct migratable : BaseComponent
    {
    private:
        typedef Mutex mutex_type;
        typedef BaseComponent base_type;
        typedef typename base_type::this_component_type this_component_type;

    public:
        migratable()
          : pin_count_(0)
        {}

        template <typename Arg>
        migratable(Arg && arg)
          : base_type(std::forward<Arg>(arg))
          , pin_count_(0)
        {}

        ~migratable()
        {
            // prevent base destructor from unregistering the gid if this
            // instance has been migrated
            if (pin_count_ == ~0x0)
                this->gid_ = naming::invalid_gid;
        }

        // Mark this component type as not migratable
        static BOOST_CONSTEXPR bool supports_migration() { return true; }

        // Pinning functionality
        void pin()
        {
            typename mutex_type::scoped_lock l(mtx_);
            ++pin_count_;
        }
        void unpin()
        {
            typename mutex_type::scoped_lock l(mtx_);
            HPX_ASSERT(pin_count_ != 0);
            if (pin_count_ != ~0x0)
                --pin_count_;
        }
        boost::uint32_t pin_count() const
        {
            typename mutex_type::scoped_lock l(mtx_);
            return pin_count_;
        }
        void mark_as_migrated()
        {
            typename mutex_type::scoped_lock l(mtx_);
            HPX_ASSERT(1 == pin_count_);
            pin_count_ = ~0x0;
        }

    private:
        mutable mutex_type mtx_;
        boost::uint32_t pin_count_;
    };
}}

#endif
