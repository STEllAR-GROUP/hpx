//  Copyright (c) 1998-2021 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/components_base/server/one_size_heap_list.hpp>
#include <hpx/components_base/server/wrapper_heap_base.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/runtime_local/state.hpp>
#include <hpx/thread_support/unlock_guard.hpp>
#include <hpx/threading_base/register_thread.hpp>
#include <hpx/threading_base/thread_data.hpp>
#if defined(HPX_DEBUG)
#include <hpx/modules/logging.hpp>
#endif

#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <string>

namespace hpx { namespace util {

    one_size_heap_list::~one_size_heap_list() noexcept
    {
#if defined(HPX_DEBUG)
        LOSH_(info).format(
            "{1}::~{1}: size({2}), max_count({3}), alloc_count({4}), "
            "free_count({5})",
            name(), heap_count_, max_alloc_count_, alloc_count_, free_count_);

        if (alloc_count_ > free_count_)
        {
            LOSH_(warning).format(
                "{1}::~{1}: releasing with {2} allocated objects", name(),
                alloc_count_ - free_count_);
        }
#endif
    }

    void* one_size_heap_list::alloc(std::size_t count)
    {
        if (HPX_UNLIKELY(0 == count))
        {
            HPX_THROW_EXCEPTION(hpx::error::bad_parameter, name() + "::alloc",
                "cannot allocate 0 objects");
        }

        void* p = nullptr;

        pthread_rwlock_rdlock(&rwlock);

        if (!heap_list_.empty())
        {
            for (auto& heap : heap_list_)
            {
                bool allocated = heap->alloc(&p, count);

                if (allocated)
                {
#if defined(HPX_DEBUG)
                    // Allocation succeeded, update statistics.
                    alloc_count_ += count;
                    if (alloc_count_ - free_count_ > max_alloc_count_)
                        max_alloc_count_ = alloc_count_ - free_count_;
#endif
                    pthread_rwlock_unlock(&rwlock);
                    return p;
                }

#if defined(HPX_DEBUG)
                LOSH_(info).format(
                    "{1}::alloc: failed to allocate from heap[{2}] "
                    "(heap[{2}] has allocated {3} objects and has "
                    "space for {4} more objects)",
                    name(), heap->heap_count(), heap->size(),
                    heap->free_size());
#endif
            }
        }
        pthread_rwlock_unlock(&rwlock);

        // Create new heap.
        bool result = false;
        std::shared_ptr<util::wrapper_heap_base> heap;
#if defined(HPX_DEBUG)
        heap = create_heap_(class_name_.c_str(), heap_count_ + 1, parameters_);
#else
        heap = create_heap_(class_name_.c_str(), 0, parameters_);
#endif
        result = heap->alloc((void**) &p, count);

        // Add the heap into the list
//        mtx_.lock();
        pthread_rwlock_wrlock(&rwlock);
        heap_list_.push_front(heap);
        pthread_rwlock_unlock(&rwlock);
//        mtx_.unlock();

        if (HPX_UNLIKELY(!result || nullptr == p))
        {
            // out of memory
            HPX_THROW_EXCEPTION(hpx::error::out_of_memory, name() + "::alloc",
                "new heap failed to allocate {1} objects", count);
        }

#if defined(HPX_DEBUG)
        alloc_count_ += count;
        ++heap_count_;

        LOSH_(info).format(
            "{1}::alloc: creating new heap[{2}], size is now {3}", name(),
            heap_count_, heap_list_.size());
#endif

        return p;
    }

    bool one_size_heap_list::reschedule(void* p, std::size_t count)
    {
        if (nullptr == threads::get_self_ptr())
        {
            hpx::threads::thread_init_data data(
                hpx::threads::make_thread_function_nullary(
                    hpx::bind_front(&one_size_heap_list::free, this, p, count)),
                "one_size_heap_list::free");
            hpx::threads::register_work(data);
            return true;
        }
        return false;
    }

    void one_size_heap_list::free(void* p, std::size_t count)
    {
        if (nullptr == p || !threads::threadmanager_is(hpx::state::running))
        {
            return;
        }

        // if this is called from outside a HPX thread we need to
        // re-schedule the request
        if (reschedule(p, count))
            return;

//        mtx_.lock();
        pthread_rwlock_rdlock(&rwlock);
//        mtx_.unlock();
        // Find the heap which allocated this pointer.
        for (auto& heap : heap_list_)
        {
            bool did_allocate = heap->did_alloc(p);
            if (did_allocate)
            {
                heap->free(p, count);
#if defined(HPX_DEBUG)
                free_count_ += count;
#endif
                pthread_rwlock_unlock(&rwlock);
                return;
            }
        }
        pthread_rwlock_unlock(&rwlock);

        HPX_THROW_EXCEPTION(hpx::error::bad_parameter, name() + "::free",
            "pointer {1} was not allocated by this {2}", p, name());
    }

    bool one_size_heap_list::did_alloc(void* p) const
    {
        pthread_rwlock_rdlock(&rwlock);
        for (typename list_type::value_type const& heap : heap_list_)
        {
            if (heap->did_alloc(p))
            {
                pthread_rwlock_unlock(&rwlock);
                return true;
            }
        }
        pthread_rwlock_unlock(&rwlock);
        return false;
    }

    std::string one_size_heap_list::name() const
    {
        if (class_name_.empty())
        {
            return std::string("one_size_heap_list(unknown)");
        }
        return std::string("one_size_heap_list(") + class_name_ + ")";
    }
}}    // namespace hpx::util
