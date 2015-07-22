//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/reinitializable_static.hpp>
#include <hpx/util/static_reinit.hpp>
#include <hpx/util/static.hpp>
#include <hpx/util/spinlock.hpp>

#include <boost/thread/locks.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    struct reinit_functions_storage
    {
        // Use util::spinlock instead of lcos::local::spinlock to avoid possible
        // suspensions of HPX threads as this will cause a deadlock when the
        // register_functions function is called from within boost::call_once
        typedef util::spinlock mutex_type;

        typedef util::function_nonser<void()> construct_type;
        typedef util::function_nonser<void()> destruct_type;

        typedef std::pair<construct_type, destruct_type> value_type;
        typedef std::vector<value_type> reinit_functions_type;

        void register_functions(construct_type const& construct,
            destruct_type const& destruct)
        {
            boost::lock_guard<mutex_type> l(mtx_);
            funcs_.push_back(value_type(construct, destruct));
        }

        void construct_all()
        {
            boost::lock_guard<mutex_type> l(mtx_);
            for (value_type const& val : funcs_)
            {
                val.first();
            }
        }

        void destruct_all()
        {
            boost::lock_guard<mutex_type> l(mtx_);
            for (value_type const& val : funcs_)
            {
                val.second();
            }
        }

        struct storage_tag {};
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
    // been terminated. This is used to initialize/reinitialize all
    // singleton instances.
    void reinit_register(util::function_nonser<void()> const& construct,
        util::function_nonser<void()> const& destruct)
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
}}
