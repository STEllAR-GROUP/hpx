//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/concurrency/spinlock.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/type_support.hpp>
#include <hpx/static_reinit/reinitializable_static.hpp>
#include <hpx/static_reinit/static_reinit.hpp>

#include <mutex>
#include <utility>
#include <vector>

namespace hpx::util {

    ///////////////////////////////////////////////////////////////////////////
    struct reinit_functions_storage
    {
        // Use util::spinlock instead of hpx::spinlock to avoid possible
        // suspensions of HPX threads as this will cause a deadlock when the
        // register_functions function is called from within std::call_once
        using mutex_type = util::spinlock;

        using construct_type = hpx::function<void()>;
        using destruct_type = hpx::function<void()>;

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
    void reinit_register(hpx::function<void()> const& construct,
        hpx::function<void()> const& destruct)
    {
        reinit_functions_storage::get().register_functions(construct, destruct);
    }

    // Invoke all globally registered construction functions
    void reinit_construct()
    {
        reinit_functions_storage::get().construct_all();
    }

    // Invoke all globally registered destruction functions
    void reinit_destruct()
    {
        reinit_functions_storage::get().destruct_all();
    }
}    // namespace hpx::util
