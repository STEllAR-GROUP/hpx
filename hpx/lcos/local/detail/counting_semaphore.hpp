//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_DETAIL_COUNTING_SEMAPHORE_AUG_03_2015_0657PM)
#define HPX_LCOS_DETAIL_COUNTING_SEMAPHORE_AUG_03_2015_0657PM

#include <hpx/config.hpp>
#include <hpx/lcos/local/detail/condition_variable.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/assert_owns_lock.hpp>

#include <boost/cstdint.hpp>
#include <boost/thread/locks.hpp>

#include <algorithm>

#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable: 4251)
#endif

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local { namespace detail
{
    class counting_semaphore
    {
    public:
        counting_semaphore(boost::int64_t value = 0)
          : value_(value)
        {}

        template <typename Mutex>
        void wait(boost::unique_lock<Mutex>& l, boost::int64_t count)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            while (value_ < count)
            {
                cond_.wait(l, "counting_semaphore::wait");
            }
            value_ -= count;
        }

        template <typename Mutex>
        bool try_wait(boost::unique_lock<Mutex>& l, boost::int64_t count = 1)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            if (!(value_ < count)) {
                // enter wait_locked only if there are sufficient credits
                // available
                wait(l, count);
                return true;
            }
            return false;
        }

        template <typename Mutex>
        void signal(boost::unique_lock<Mutex> l, boost::int64_t count)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            Mutex* mtx = l.mutex();

            // release no more threads than we get resources
            value_ += count;
            for (boost::int64_t i = 0; value_ >= 0 && i < count; ++i)
            {
                // notify_one() returns false if no more threads are
                // waiting
                if (!cond_.notify_one(std::move(l)))
                    break;

                l = boost::unique_lock<Mutex>(*mtx);
            }
        }

        template <typename Mutex>
        boost::int64_t signal_all(boost::unique_lock<Mutex> l)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            boost::int64_t count = static_cast<boost::int64_t>(cond_.size(l));
            signal(std::move(l), count);
            return count;
        }

    private:
        boost::int64_t value_;
        local::detail::condition_variable cond_;
    };
}}}}

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif

#endif

