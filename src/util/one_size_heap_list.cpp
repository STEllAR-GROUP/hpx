//  Copyright (c) 1998-2017 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/state.hpp>
#include <hpx/util/one_size_heap_list.hpp>
#if defined(HPX_DEBUG)
#include <hpx/modules/logging.hpp>
#endif
#include <hpx/thread_support/unlock_guard.hpp>
#include <hpx/threading_base/register_thread.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/util/wrapper_heap_base.hpp>

#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <string>

namespace hpx { namespace util {
    one_size_heap_list::~one_size_heap_list() noexcept
    {
#if defined(HPX_DEBUG)
        LOSH_(info) << hpx::util::format(
            "{1}::~{1}: size({2}), max_count({3}), alloc_count({4}), "
            "free_count({5})",
            name(), heap_count_, max_alloc_count_, alloc_count_, free_count_);

        if (alloc_count_ > free_count_)
        {
            LOSH_(warning) << hpx::util::format(
                "{1}::~{1}: releasing with {2} allocated objects", name(),
                alloc_count_ - free_count_);
        }
#endif
    }

    void* one_size_heap_list::alloc(std::size_t count)
    {
        unique_lock_type guard(mtx_);

        if (HPX_UNLIKELY(0 == count))
        {
            guard.unlock();
            HPX_THROW_EXCEPTION(
                bad_parameter, name() + "::alloc", "cannot allocate 0 objects");
        }

        //std::size_t size = 0;
        void* p = nullptr;
        {
            if (!heap_list_.empty())
            {
                //size = heap_list_.size();
                for (auto& heap : heap_list_)
                {
                    bool allocated = false;

                    {
                        util::unlock_guard<unique_lock_type> ul(guard);
                        allocated = heap->alloc(&p, count);
                    }

                    if (allocated)
                    {
#if defined(HPX_DEBUG)
                        // Allocation succeeded, update statistics.
                        alloc_count_ += count;
                        if (alloc_count_ - free_count_ > max_alloc_count_)
                            max_alloc_count_ = alloc_count_ - free_count_;
#endif
                        return p;
                    }

#if defined(HPX_DEBUG)
                    LOSH_(info) << hpx::util::format(
                        "{1}::alloc: failed to allocate from heap[{2}] "
                        "(heap[{2}] has allocated {3} objects and has "
                        "space for {4} more objects)",
                        name(), heap->heap_count(), heap->size(),
                        heap->free_size());
#endif
                }
            }
        }

        // Create new heap.
        bool did_create = false;
        {
#if defined(HPX_DEBUG)
            heap_list_.push_front(create_heap_(
                class_name_.c_str(), heap_count_ + 1, parameters_));
#else
            heap_list_.push_front(
                create_heap_(class_name_.c_str(), 0, parameters_));
#endif

            iterator itnew = heap_list_.begin();
            typename list_type::value_type heap = *itnew;
            bool result = false;

            {
                util::unlock_guard<unique_lock_type> ul(guard);
                result = heap->alloc((void**) &p, count);
            }

            if (HPX_UNLIKELY(!result || nullptr == p))
            {
                // out of memory
                guard.unlock();
                HPX_THROW_EXCEPTION(out_of_memory, name() + "::alloc",
                    hpx::util::format(
                        "new heap failed to allocate {1} objects", count));
            }

#if defined(HPX_DEBUG)
            alloc_count_ += count;
            ++heap_count_;

            LOSH_(info) << hpx::util::format(
                "{1}::alloc: creating new heap[{2}], size is now {3}", name(),
                heap_count_, heap_list_.size());
#endif
            did_create = true;
        }

        if (did_create)
            return p;

        guard.unlock();

        // Try again, we just got a new heap, so we should be good.
        return alloc(count);
    }

    bool one_size_heap_list::reschedule(void* p, std::size_t count)
    {
        if (nullptr == threads::get_self_ptr())
        {
            hpx::threads::thread_init_data data(
                hpx::threads::make_thread_function_nullary(util::bind_front(
                    &one_size_heap_list::free, this, p, count)),
                "one_size_heap_list::free");
            hpx::threads::register_work(data);
            return true;
        }
        return false;
    }

    void one_size_heap_list::free(void* p, std::size_t count)
    {
        unique_lock_type ul(mtx_);

        if (nullptr == p || !threads::threadmanager_is(state_running))
            return;

        // if this is called from outside a HPX thread we need to
        // re-schedule the request
        if (reschedule(p, count))
            return;

        // Find the heap which allocated this pointer.
        for (auto& heap : heap_list_)
        {
            bool did_allocate = false;

            {
                util::unlock_guard<unique_lock_type> ull(ul);
                did_allocate = heap->did_alloc(p);
                if (did_allocate)
                    heap->free(p, count);
            }

            if (did_allocate)
            {
#if defined(HPX_DEBUG)
                free_count_ += count;
#endif
                return;
            }
        }

        ul.unlock();

        HPX_THROW_EXCEPTION(bad_parameter, name() + "::free",
            hpx::util::format(
                "pointer {1} was not allocated by this {2}", p, name()));
    }

    bool one_size_heap_list::did_alloc(void* p) const
    {
        unique_lock_type ul(mtx_);
        for (typename list_type::value_type const& heap : heap_list_)
        {
            bool did_allocate = false;

            {
                util::unlock_guard<unique_lock_type> ull(ul);
                did_allocate = heap->did_alloc(p);
            }

            if (did_allocate)
                return true;
        }
        return false;
    }

    std::string one_size_heap_list::name() const
    {
        if (class_name_.empty())
            return std::string("one_size_heap_list(unknown)");
        return std::string("one_size_heap_list(") + class_name_ + ")";
    }
}}    // namespace hpx::util
