//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_CANCELLATION_TOKEN_OCT_05_1205PM)
#define HPX_PARALLEL_UTIL_CANCELLATION_TOKEN_OCT_05_1205PM

#include <hpx/config.hpp>

#include <boost/atomic.hpp>

#include <algorithm>
#include <functional>
#include <memory>

namespace hpx { namespace parallel { namespace util
{
    namespace detail
    {
        struct no_data
        {
            bool operator<= (no_data) const { return true; }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // cancellation_token is used for premature cancellation of algorithms
    template <typename T = detail::no_data, typename Pred = std::less_equal<T> >
    class cancellation_token
    {
    private:
        typedef boost::atomic<T> flag_type;
        std::shared_ptr<flag_type> was_cancelled_;

    public:
        cancellation_token(T data)
          : was_cancelled_(std::make_shared<flag_type>(data))
        {}

        bool was_cancelled(T data) const noexcept
        {
            return Pred()(was_cancelled_->load(boost::memory_order_relaxed), data);
        }

        void cancel(T data) noexcept
        {
            T old_data = was_cancelled_->load(boost::memory_order_relaxed);

            do {
                if (Pred()(old_data, data))
                    break;      // if we already have a closer one, break

            } while (!was_cancelled_->compare_exchange_strong(old_data, data,
                boost::memory_order_relaxed));
        }

        T get_data() const noexcept
        {
            return was_cancelled_->load(boost::memory_order_relaxed);
        }
    };

    // special case for when no additional data needs to be stored at the
    // cancellation point
    template <>
    class cancellation_token<detail::no_data, std::less_equal<detail::no_data> >
    {
    private:
        typedef boost::atomic<bool> flag_type;
        std::shared_ptr<flag_type> was_cancelled_;

    public:
        cancellation_token()
          : was_cancelled_(std::make_shared<flag_type>(false))
        {}

        bool was_cancelled() const noexcept
        {
            return was_cancelled_->load(boost::memory_order_relaxed);
        }

        void cancel() noexcept
        {
            was_cancelled_->store(true, boost::memory_order_relaxed);
        }
    };
}}}

#endif
