//  Copyright (c) 2007-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/concurrency/spinlock.hpp>
#include <hpx/static_reinit/reinitializable_static.hpp>
#include <hpx/static_reinit/static_reinit.hpp>
#include <hpx/type_support/detail/static_reinit_functions.hpp>
#include <hpx/type_support/static.hpp>

#include <functional>
#include <mutex>
#include <utility>
#include <vector>

namespace hpx::util::detail {

    ///////////////////////////////////////////////////////////////////////////
    struct reinit_functions_storage
    {
        // Use util::spinlock instead of hpx::spinlock to avoid possible
        // suspensions of HPX threads as this will cause a deadlock when the
        // register_functions function is called from within std::call_once
        using mutex_type = util::spinlock;

        using construct_type = std::function<void()>;
        using destruct_type = std::function<void()>;

        using value_type = std::pair<construct_type, destruct_type>;
        using reinit_functions_type = std::vector<value_type>;

        void register_functions(
            construct_type const& construct, destruct_type const& destruct)
        {
            std::lock_guard<mutex_type> l(mtx_);
            funcs_.emplace_back(construct, destruct);
        }

        void construct_all()
        {
            std::lock_guard<mutex_type> l(mtx_);
            for (value_type const& val : funcs_)
            {
                val.first();
            }
        }

        void destruct_all()
        {
            std::lock_guard<mutex_type> l(mtx_);
            for (value_type const& val : funcs_)
            {
                val.second();
            }
        }

        struct storage_tag
        {
        };
        static reinit_functions_storage& get();

    private:
        reinit_functions_type funcs_;
        mutex_type mtx_;
    };

    inline reinit_functions_storage& reinit_functions_storage::get()
    {
        util::static_<reinit_functions_storage, storage_tag> storage;
        return storage.get();
    }

    // This is a global API allowing to register functions to be called before
    // the runtime system is about to start and after the runtime system has
    // been terminated. This is used to initialize/reinitialize all singleton
    // instances.
    void reinit_register_impl(std::function<void()> const& construct,
        std::function<void()> const& destruct)
    {
        reinit_functions_storage::get().register_functions(construct, destruct);
    }

    // Invoke all globally registered construction functions
    void reinit_construct_impl()
    {
        reinit_functions_storage::get().construct_all();
    }

    // Invoke all globally registered destruction functions
    void reinit_destruct_impl()
    {
        reinit_functions_storage::get().destruct_all();
    }
}    // namespace hpx::util::detail

namespace hpx::util {

    // initialize AGAS interface function pointers in components_base module
    struct HPX_CORE_EXPORT static_reinit_interface_functions
    {
        static_reinit_interface_functions()
        {
            detail::reinit_register = &detail::reinit_register_impl;
            detail::reinit_construct = &detail::reinit_construct_impl;
            detail::reinit_destruct = &detail::reinit_destruct_impl;
        }
    };

    static_reinit_interface_functions& static_reinit_init()
    {
        static static_reinit_interface_functions static_reinit_init_;
        return static_reinit_init_;
    }
}    // namespace hpx::util
