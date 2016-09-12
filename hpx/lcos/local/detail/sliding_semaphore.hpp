//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_DETAIL_SLIDING_SEMAPHORE_AUG_25_2016_1026AM)
#define HPX_LCOS_DETAIL_SLIDING_SEMAPHORE_AUG_25_2016_1026AM

#include <hpx/config.hpp>
#include <hpx/lcos/local/detail/condition_variable.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/assert_owns_lock.hpp>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <utility>

#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable: 4251)
#endif

////////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace local { namespace detail
{
    class sliding_semaphore
    {
    private:
        typedef lcos::local::spinlock mutex_type;

    public:
        sliding_semaphore(std::int64_t max_difference, std::int64_t lower_limit)
          : max_difference_(max_difference), lower_limit_(lower_limit), cond_()
        {}

        void wait(std::unique_lock<mutex_type>& l, std::int64_t upper_limit)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            while (upper_limit - max_difference_ > lower_limit_)
            {
                cond_.wait(l, "sliding_semaphore::wait");
            }
        }

        bool try_wait(std::unique_lock<mutex_type>& l, std::int64_t upper_limit)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            if (!(upper_limit - max_difference_ > lower_limit_))
            {
                // enter wait_locked only if necessary
                wait(l, upper_limit);
                return true;
            }
            return false;
        }

        void signal(std::unique_lock<mutex_type> l, std::int64_t lower_limit)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            mutex_type* mtx = l.mutex();

            // touch upon all threads
            std::int64_t count = static_cast<std::int64_t>(cond_.size(l));
            for (/**/; count > 0; --count)
            {
                // notify_one() returns false if no more threads are waiting
                if (!cond_.notify_one(std::move(l)))
                    break;

                l = std::unique_lock<mutex_type>(*mtx);
            }
            lower_limit_ = (std::max)(lower_limit, lower_limit_);
        }

        std::int64_t signal_all(std::unique_lock<mutex_type> l)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            signal(std::move(l), lower_limit_);
            return lower_limit_;
        }

    private:
        std::int64_t max_difference_;
        std::int64_t lower_limit_;
        local::detail::condition_variable cond_;
    };
}}}}

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif

#endif

