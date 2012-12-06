//  Copyright (c) 1998-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_041EF599_BA27_47ED_B1F0_2691B28966B3)
#define HPX_041EF599_BA27_47ED_B1F0_2691B28966B3

#include <list>
#include <string>

#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/atomic.hpp>
#include <boost/format.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/type_traits/is_same.hpp>

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/state.hpp>
#include <hpx/exception.hpp>
#include <hpx/lcos/local/shared_mutex.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/bind.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    template <typename Heap, typename SharedMutex =
        lcos::local::detail::shared_mutex<lcos::local::spinlock> >
    class one_size_heap_list
    {
    public:
        typedef Heap heap_type;

        typedef typename heap_type::allocator_type allocator_type;
        typedef typename heap_type::value_type value_type;

        typedef std::list<boost::shared_ptr<heap_type> > list_type;
        typedef typename list_type::iterator iterator;
        typedef typename list_type::const_iterator const_iterator;

        enum
        {
            heap_step = heap_type::heap_step,   // default grow step
            heap_size = heap_type::heap_size    // size of the object
        };

        typedef SharedMutex mutex_type;

        typedef boost::shared_lock<mutex_type> shared_lock_type;
        typedef boost::upgrade_lock<mutex_type> upgrade_lock_type;
        typedef boost::upgrade_to_unique_lock<mutex_type> upgraded_lock_type;
        typedef boost::unique_lock<mutex_type> unique_lock_type;

        explicit one_size_heap_list(char const* class_name = "")
            : class_name_(class_name)
            , alloc_count_(0L)
            , free_count_(0L)
            , heap_count_(0L)
            , max_alloc_count_(0L)
        {
            BOOST_ASSERT(sizeof(typename heap_type::storage_type) == heap_size);
        }

        explicit one_size_heap_list(std::string const& class_name)
            : class_name_(class_name)
            , alloc_count_(0L)
            , free_count_(0L)
            , heap_count_(0L)
            , max_alloc_count_(0L)
        {
            BOOST_ASSERT(sizeof(typename heap_type::storage_type) == heap_size);
        }

        ~one_size_heap_list()
        {
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
        }

        // operations
        value_type* alloc(std::size_t count = 1)
        {
            if (HPX_UNLIKELY(0 == count))
            {
                HPX_THROW_EXCEPTION(bad_parameter,
                    name() + "::alloc",
                    "cannot allocate 0 objects");
            }

            std::size_t size = 0;
            value_type* p = NULL;
            {
                shared_lock_type guard(mtx_);

                if (!heap_list_.empty())
                {
                    size = heap_list_.size();
                    for (iterator it = heap_list_.begin(); it != heap_list_.end(); ++it)
                    {
                        if ((*it)->alloc(&p, count))
                        {
                            // Allocation succeeded, update statistics.
                            alloc_count_ += count;

                            if (alloc_count_ - free_count_ > max_alloc_count_)
                                max_alloc_count_ = alloc_count_- free_count_;

                            return p;
                        }

                        LOSH_(info)
                            << (boost::format(
                                "%1%::alloc: failed to allocate from heap[%2%] "
                                "(heap[%2%] has allocated %3% objects and has "
                                "space for %4% more objects)")
                                % name()
                                % (*it)->heap_count_
                                % (*it)->size()
                                % (*it)->free_size());
                    }
                }
            }

            // Create new heap.
            bool did_create = false;
            {
                // Acquire exclusive access.
                unique_lock_type ul(mtx_);

                heap_list_.push_front(typename list_type::value_type(
                    new heap_type(class_name_.c_str(), heap_count_ + 1, heap_step)));

                iterator itnew = heap_list_.begin();
                bool result = (*itnew)->alloc(&p, count);

                if (HPX_UNLIKELY(!result || NULL == p))
                {
                    // out of memory
                    HPX_THROW_EXCEPTION(out_of_memory,
                        name() + "::alloc",
                        boost::str(boost::format(
                            "new heap failed to allocate %1% objects")
                            % count));
                }

                alloc_count_ += count;
                ++heap_count_;

                LOSH_(info)
                    << (boost::format(
                        "%1%::alloc: creating new heap[%2%], size is now %3%")
                        % name()
                        % heap_count_
                        % size);
                did_create = true;
            }

            if (did_create)
                return p;

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
                    name() + "::add_heap", "encountered NULL heap");
            }

            // Acquire exclusive access.
            unique_lock_type ul(mtx_);

            p->heap_count_ = heap_count_;

            iterator it = heap_list_.insert(heap_list_.begin(),
                typename list_type::value_type(p));

            // Check for insertion failure.
            if (HPX_UNLIKELY(it == heap_list_.end()))
            {
                HPX_THROW_EXCEPTION(out_of_memory,
                    name() + "::add_heap",
                    boost::str(boost::format("heap %1% could not be added") % p));
            }

            ++heap_count_;
        }

        // need to reschedule if not using boost::shared_mutex
        bool reschedule(void* p, std::size_t count, boost::mpl::false_)
        {
            if (0 == threads::get_self_ptr())
            {
                hpx::applier::register_work(
                    util::bind(&one_size_heap_list::free, this, p, count),
                    "one_size_heap_list::free");
                return true;
            }
            return false;
        }

        bool reschedule(void* p, std::size_t count, boost::mpl::true_)
        {
            return false;
        }

        void free(void* p, std::size_t count = 1)
        {
            if (NULL == p || !threads::threadmanager_is(running))
                return;

            // if this is called from outside a HPX thread we need to
            // re-schedule the request
            typedef boost::is_same<boost::shared_mutex, SharedMutex>
                reschedule_pred;
            if (reschedule(p, count, reschedule_pred()))
                return;

            shared_lock_type guard(mtx_);

            // Find the heap which allocated this pointer.
            for (iterator it = heap_list_.begin(); it != heap_list_.end(); ++it)
            {
                if ((*it)->did_alloc(p))
                {
                    (*it)->free(p, count);
                    free_count_ += count;
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
            shared_lock_type guard(mtx_);
            for (iterator it = heap_list_.begin(); it != heap_list_.end(); ++it)
            {
                if ((*it)->did_alloc(p))
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
        mutex_type mtx_;
        list_type heap_list_;

    private:
        std::string const class_name_;

    public:
        std::size_t alloc_count_;
        std::size_t free_count_;
        std::size_t heap_count_;
        std::size_t max_alloc_count_;
    };
}} // namespace hpx::util

#endif // HPX_041EF599_BA27_47ED_B1F0_2691B28966B3

