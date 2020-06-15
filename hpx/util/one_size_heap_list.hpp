//  Copyright (c) 1998-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/util/wrapper_heap_base.hpp>

#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <string>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    class HPX_EXPORT one_size_heap_list
    {
    public:
        typedef std::list<std::shared_ptr<util::wrapper_heap_base> > list_type;
        typedef typename list_type::iterator iterator;
        typedef typename list_type::const_iterator const_iterator;

        typedef lcos::local::spinlock mutex_type;

        typedef std::unique_lock<mutex_type> unique_lock_type;

        typedef wrapper_heap_base::heap_parameters heap_parameters;

    private:
        template <typename Heap>
        static std::shared_ptr<util::wrapper_heap_base> create_heap(
            char const* name, std::size_t counter, heap_parameters parameters)
        {
#if defined(HPX_DEBUG)
            return std::make_shared<Heap>(name, counter, parameters);
#else
            (void)counter;
            return std::make_shared<Heap>(name, 0, parameters);
#endif
        }

    public:
        one_size_heap_list()
            : class_name_()
#if defined(HPX_DEBUG)
            , alloc_count_(0)
            , free_count_(0)
            , heap_count_(0)
            , max_alloc_count_(0)
#endif
            , create_heap_(nullptr)
            , parameters_({0, 0, 0})
        {
            HPX_ASSERT(false); // shouldn't ever be called
        }

        template <typename Heap>
        explicit one_size_heap_list(char const* class_name,
                heap_parameters parameters, Heap* = nullptr)
            : class_name_(class_name)
#if defined(HPX_DEBUG)
            , alloc_count_(0L)
            , free_count_(0L)
            , heap_count_(0L)
            , max_alloc_count_(0L)
#endif
            , create_heap_(&one_size_heap_list::create_heap<Heap>)
            , parameters_(parameters)
        {}

        template <typename Heap>
        explicit one_size_heap_list(std::string const& class_name,
                heap_parameters parameters, Heap* = nullptr)
            : class_name_(class_name)
#if defined(HPX_DEBUG)
            , alloc_count_(0L)
            , free_count_(0L)
            , heap_count_(0L)
            , max_alloc_count_(0L)
#endif
            , create_heap_(&one_size_heap_list::create_heap<Heap>)
            , parameters_(parameters)
        {}

        ~one_size_heap_list() noexcept;

        // operations
        void* alloc(std::size_t count = 1);

        // need to reschedule if not using boost::mutex
        bool reschedule(void* p, std::size_t count);

        void free(void* p, std::size_t count = 1);

        bool did_alloc(void* p) const;

        std::string name() const;

    protected:
        mutable mutex_type mtx_;
        list_type heap_list_;

    private:
        std::string const class_name_;

    public:
#if defined(HPX_DEBUG)
        std::size_t alloc_count_;
        std::size_t free_count_;
        std::size_t heap_count_;
        std::size_t max_alloc_count_;
#endif
        std::shared_ptr<util::wrapper_heap_base> (*create_heap_)(
            char const*, std::size_t, heap_parameters);

        heap_parameters const parameters_;
    };
}}

#include <hpx/config/warnings_suffix.hpp>

