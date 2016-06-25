//  Copyright (c) 1998-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_ONE_SIZE_HEAP_LIST_HPP
#define HPX_UTIL_ONE_SIZE_HEAP_LIST_HPP

#include <hpx/config.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/bind.hpp>
#if defined(HPX_DEBUG)
#include <hpx/util/logging.hpp>
#endif
#include <hpx/util/one_size_heap_list_base.hpp>
#include <hpx/util/unlock_guard.hpp>

#include <boost/format.hpp>

#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    template <typename Heap, typename Mutex = lcos::local::spinlock>
    class one_size_heap_list : public one_size_heap_list_base
    {
    public:
        typedef Heap heap_type;

        typedef typename heap_type::allocator_type allocator_type;
        typedef typename heap_type::value_type value_type;

        typedef std::list<std::shared_ptr<heap_type> > list_type;
        typedef typename list_type::iterator iterator;
        typedef typename list_type::const_iterator const_iterator;

        enum
        {
            heap_step = Heap::heap_step,   // default grow step
            heap_size = Heap::heap_size    // size of the object
        };

        typedef Mutex mutex_type;

        typedef std::unique_lock<mutex_type> unique_lock_type;

        explicit one_size_heap_list(char const* class_name = "")
            : class_name_(class_name)
#if defined(HPX_DEBUG)
            , alloc_count_(0L)
            , free_count_(0L)
            , heap_count_(0L)
            , max_alloc_count_(0L)
#endif
        {
            HPX_ASSERT(sizeof(typename heap_type::storage_type) == uint64_t(heap_size));
        }

        explicit one_size_heap_list(std::string const& class_name)
            : class_name_(class_name)
#if defined(HPX_DEBUG)
            , alloc_count_(0L)
            , free_count_(0L)
            , heap_count_(0L)
            , max_alloc_count_(0L)
#endif
        {
            HPX_ASSERT(sizeof(typename heap_type::storage_type) == uint64_t(heap_size));
        }

        ~one_size_heap_list()
        {
#if defined(HPX_DEBUG)
            LOSH_(info)
                << (boost::format(
                   "%1%::~%1%: size(%2%), max_count(%3%), alloc_count(%4%), "
                   "free_count(%5%)")
                   % name()
                   % heap_count_
                   % max_alloc_count_
                   % alloc_count_
                   % free_count_);

            if (alloc_count_ > free_count_)
            {
                LOSH_(warning)
                    << (boost::format(
                       "%1%::~%1%: releasing with %2% allocated objects")
                       % name()
                       % (alloc_count_ - free_count_));
            }
#endif
        }

        // operations
        void* alloc(std::size_t count = 1)
        {
            unique_lock_type guard(mtx_);

            if (HPX_UNLIKELY(0 == count))
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    name() + "::alloc",
                    "cannot allocate 0 objects");
            }

            //std::size_t size = 0;
            value_type* p = nullptr;
            {
                if (!heap_list_.empty())
                {
                    //size = heap_list_.size();
                    for (iterator it = heap_list_.begin(); it != heap_list_.end(); ++it)
                    {
                        typename list_type::value_type heap = *it;
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
                                max_alloc_count_ = alloc_count_- free_count_;
#endif
                            return p;
                        }

#if defined(HPX_DEBUG)
                        LOSH_(info)
                            << (boost::format(
                                "%1%::alloc: failed to allocate from heap[%2%] "
                                "(heap[%2%] has allocated %3% objects and has "
                                "space for %4% more objects)")
                                % name()
                                % (*it)->heap_count_
                                % (*it)->size()
                                % (*it)->free_size());
#endif
                    }
                }
            }

            // Create new heap.
            bool did_create = false;
            {
#if defined(HPX_DEBUG)
                heap_list_.push_front(typename list_type::value_type(
                    new heap_type(class_name_.c_str(), heap_count_ + 1, heap_step)));
#else
                heap_list_.push_front(typename list_type::value_type(
                    new heap_type(class_name_.c_str(), 0, heap_step)));
#endif

                iterator itnew = heap_list_.begin();
                typename list_type::value_type heap = *itnew;
                bool result = false;

                {
                    util::unlock_guard<unique_lock_type> ul(guard);
                    result = heap->alloc(&p, count);
                }

                if (HPX_UNLIKELY(!result || nullptr == p))
                {
                    // out of memory
                    HPX_THROW_EXCEPTION(out_of_memory,
                        name() + "::alloc",
                        boost::str(boost::format(
                            "new heap failed to allocate %1% objects")
                            % count));
                }

#if defined(HPX_DEBUG)
                alloc_count_ += count;
                ++heap_count_;

                LOSH_(info)
                    << (boost::format(
                        "%1%::alloc: creating new heap[%2%], size is now %3%")
                        % name()
                        % heap_count_
                        % heap_list_.size());
#endif
                did_create = true;
            }

            if (did_create)
                return p;

            guard.unlock();

            // Try again, we just got a new heap, so we should be good.
            return alloc(count);
        }

        heap_type* alloc_heap()
        {
            return new heap_type(class_name_.c_str(), 0, heap_step);
        }

        void add_heap(heap_type* p)
        {
            if (HPX_UNLIKELY(!p))
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    name() + "::add_heap", "encountered nullptr heap");
            }

            unique_lock_type ul(mtx_);
#if defined(HPX_DEBUG)
            p->heap_count_ = heap_count_;
#endif

            iterator it = heap_list_.insert(heap_list_.begin(),
                typename list_type::value_type(p));

            // Check for insertion failure.
            if (HPX_UNLIKELY(it == heap_list_.end()))
            {
                HPX_THROW_EXCEPTION(out_of_memory,
                    name() + "::add_heap",
                    boost::str(boost::format("heap %1% could not be added") % p));
            }

#if defined(HPX_DEBUG)
            ++heap_count_;
#endif
        }

        // need to reschedule if not using boost::mutex
        bool reschedule(void* p, std::size_t count)
        {
            if (nullptr == threads::get_self_ptr())
            {
                hpx::applier::register_work(
                    util::bind(&one_size_heap_list::free, this, p, count),
                    "one_size_heap_list::free");
                return true;
            }
            return false;
        }

        void free(void* p, std::size_t count = 1)
        {
            unique_lock_type ul(mtx_);

            if (nullptr == p || !threads::threadmanager_is(state_running))
                return;

            // if this is called from outside a HPX thread we need to
            // re-schedule the request
            if (reschedule(p, count))
                return;

            // Find the heap which allocated this pointer.
            for (iterator it = heap_list_.begin(); it != heap_list_.end(); ++it)
            {
                typename list_type::value_type heap = *it;
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

            HPX_THROW_EXCEPTION(bad_parameter,
                name() + "::free",
                boost::str(boost::format(
                    "pointer %1% was not allocated by this %2%")
                    % p % name()));
        }

        bool did_alloc(void* p) const
        {
            unique_lock_type ul(mtx_);
            for (const_iterator it = heap_list_.begin(); it != heap_list_.end(); ++it)
            {
                typename list_type::value_type heap = *it;
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

        std::string name() const
        {
            if (class_name_.empty())
                return std::string("one_size_heap_list(unknown)");
            return std::string("one_size_heap_list(") + class_name_ + ")";
        }

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
    };
}} // namespace hpx::util

#endif /*HPX_UTIL_ONE_SIZE_HEAP_LIST_HPP*/
