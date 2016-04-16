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

#include <algorithm>
#include <mutex>

#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable: 4251)
#endif

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local { namespace detail
{
    class counting_semaphore
    {
    private:
        typedef lcos::local::spinlock mutex_type;

    public:
        counting_semaphore(boost::int64_t value = 0)
          : value_(value)
        {}

        void wait(std::unique_lock<mutex_type>& l, boost::int64_t count)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            while (value_ < count)
            {
                cond_.wait(l, "counting_semaphore::wait");
            }
            value_ -= count;
        }

        bool try_wait(std::unique_lock<mutex_type>& l, boost::int64_t count = 1)
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

        void signal(std::unique_lock<mutex_type> l, boost::int64_t count)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            mutex_type* mtx = l.mutex();

            // release no more threads than we get resources
            value_ += count;
            for (boost::int64_t i = 0; value_ >= 0 && i < count; ++i)
            {
                // notify_one() returns false if no more threads are
                // waiting
                if (!cond_.notify_one(std::move(l)))
                    break;

                l = std::unique_lock<mutex_type>(*mtx);
            }
        }

        boost::int64_t signal_all(std::unique_lock<mutex_type> l)
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

