//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This code has been partially adopted from the Boost.Threads library
//
// (C) Copyright 2008 Anthony Williams
// (C) Copyright 2011-2012 Vicente J. Botet Escriba

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/coroutines/coroutine.hpp>
#include <hpx/runtime/threads/coroutines/detail/coroutine_self.hpp>
#include <hpx/runtime/threads/coroutines/detail/tss.hpp>
#include <hpx/util/assert.hpp>

#include <hpx/runtime/threads_fwd.hpp>

#include <boost/shared_ptr.hpp>

#include <map>

namespace hpx { namespace threads { namespace coroutines { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    void tss_data_node::cleanup(bool cleanup_existing)
    {
#ifdef HPX_HAVE_THREAD_LOCAL_STORAGE
        if (cleanup_existing && func_ && (value_ != 0))
        {
            (*func_)(value_);
        }
        func_.reset();
        value_ = 0;
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    tss_storage* create_tss_storage()
    {
#ifdef HPX_HAVE_THREAD_LOCAL_STORAGE
        return new tss_storage;
#else
        boost::throw_exception(std::runtime_error(
            "thread local storage has been disabled at configuration time, "
            "please specify HPX_HAVE_THREAD_LOCAL_STORAGE=ON to cmake"));
        return 0;
#endif
    }

    void delete_tss_storage(tss_storage*& storage)
    {
#ifdef HPX_HAVE_THREAD_LOCAL_STORAGE
        delete storage;
        storage = 0;
#endif
    }

    std::size_t get_tss_thread_data(tss_storage* storage)
    {
#ifdef HPX_HAVE_THREAD_LOCAL_STORAGE
        hpx::threads::thread_self* self = hpx::threads::get_self_ptr();
        if (NULL == self)
        {
            boost::throw_exception(null_thread_id_exception());
            return 0;
        }

        detail::tss_storage* tss_map = self->get_thread_tss_data();
        if (NULL == tss_map)
            return 0;

        tss_data_node* node = tss_map->find(0);
        if (0 == node)
            return 0;

        return node->get_data<std::size_t>();
#else
        boost::throw_exception(std::runtime_error(
            "thread local storage has been disabled at configuration time, "
            "please specify HPX_HAVE_THREAD_LOCAL_STORAGE=ON to cmake"));
        return 0;
#endif
    }

    std::size_t set_tss_thread_data(tss_storage* storage, std::size_t data)
    {
#ifdef HPX_HAVE_THREAD_LOCAL_STORAGE
        hpx::threads::thread_self* self = hpx::threads::get_self_ptr();
        if (NULL == self)
        {
            boost::throw_exception(null_thread_id_exception());
            return 0;
        }

        detail::tss_storage* tss_map = self->get_or_create_thread_tss_data();
        if (NULL == tss_map)
        {
            boost::throw_exception(std::bad_alloc());
            return 0;
        }

        tss_data_node* node = tss_map->find(0);
        if (0 == node)
        {
            tss_map->insert(0, new std::size_t(data)); //-V508
            return 0;
        }

        std::size_t prev_val = node->get_data<std::size_t>();
        node->set_data(data);

        return prev_val;
#else
        boost::throw_exception(std::runtime_error(
            "thread local storage has been disabled at configuration time, "
            "please specify HPX_HAVE_THREAD_LOCAL_STORAGE=ON to cmake"));
        return 0;
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    tss_data_node* find_tss_data(void const* key)
    {
#ifdef HPX_HAVE_THREAD_LOCAL_STORAGE
        hpx::threads::thread_self* self = hpx::threads::get_self_ptr();
        if (NULL == self)
        {
            boost::throw_exception(null_thread_id_exception());
            return 0;
        }

        detail::tss_storage* tss_map = self->get_thread_tss_data();
        if (NULL == tss_map)
            return 0;

        return tss_map->find(key);
#else
        boost::throw_exception(std::runtime_error(
            "thread local storage has been disabled at configuration time, "
            "please specify HPX_HAVE_THREAD_LOCAL_STORAGE=ON to cmake"));
        return 0;
#endif
    }

    void* get_tss_data(void const* key)
    {
#ifdef HPX_HAVE_THREAD_LOCAL_STORAGE
        if (tss_data_node* const current_node = find_tss_data(key))
            return current_node->get_value();
#endif
        return NULL;
    }

    void add_new_tss_node(void const* key,
        boost::shared_ptr<tss_cleanup_function> const& func, void* tss_data)
    {
#ifdef HPX_HAVE_THREAD_LOCAL_STORAGE
        hpx::threads::thread_self* self = hpx::threads::get_self_ptr();
        if (NULL == self)
        {
            boost::throw_exception(null_thread_id_exception());
            return;
        }

        detail::tss_storage* tss_map = self->get_or_create_thread_tss_data();
        if (NULL == tss_map)
        {
            boost::throw_exception(std::bad_alloc());
            return;
        }

        tss_map->insert(key, func, tss_data);
#endif
    }

    void erase_tss_node(void const* key, bool cleanup_existing)
    {
#ifdef HPX_HAVE_THREAD_LOCAL_STORAGE
        hpx::threads::thread_self* self = hpx::threads::get_self_ptr();
        if (NULL == self)
        {
            boost::throw_exception(null_thread_id_exception());
            return;
        }

        detail::tss_storage* tss_map = self->get_thread_tss_data();
        if (NULL != tss_map)
            tss_map->erase(key, cleanup_existing);
#endif
    }

    void set_tss_data(void const* key,
        boost::shared_ptr<tss_cleanup_function> const& func,
        void* tss_data, bool cleanup_existing)
    {
#ifdef HPX_HAVE_THREAD_LOCAL_STORAGE
        if (tss_data_node* const current_node = find_tss_data(key))
        {
            if (func || (tss_data != 0))
                current_node->reinit(func, tss_data, cleanup_existing);
            else
                erase_tss_node(key, cleanup_existing);
        }
        else if(func || (tss_data != 0))
        {
            add_new_tss_node(key, func, tss_data);
        }
#endif
    }
}}}}

