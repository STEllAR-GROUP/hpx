//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>

namespace hpx { namespace parallel { namespace util {
    namespace detail {
        struct no_data
        {
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // cancellation_token is used for premature cancellation of algorithms
    template <typename T = detail::no_data, typename Pred = std::less_equal<T>>
    class cancellation_token
    {
    private:
        typedef std::atomic<T> flag_type;
        std::shared_ptr<flag_type> was_cancelled_;

    public:
        cancellation_token(T data)
          : was_cancelled_(std::make_shared<flag_type>(data))
        {
        }

        bool was_cancelled(T data) const noexcept
        {
            return Pred()(
                was_cancelled_->load(std::memory_order_relaxed), data);
        }

        void cancel(T data) noexcept
        {
            T old_data = was_cancelled_->load(std::memory_order_relaxed);

            do
            {
                if (Pred()(old_data, data))
                    break;    // if we already have a closer one, break

            } while (!was_cancelled_->compare_exchange_strong(
                old_data, data, std::memory_order_relaxed));
        }

        T get_data() const noexcept
        {
            return was_cancelled_->load(std::memory_order_relaxed);
        }
    };

    // special case for when no additional data needs to be stored at the
    // cancellation point
    template <>
    class cancellation_token<detail::no_data, std::less_equal<detail::no_data>>
    {
    private:
        typedef std::atomic<bool> flag_type;
        std::shared_ptr<flag_type> was_cancelled_;

    public:
        cancellation_token()
          : was_cancelled_(std::make_shared<flag_type>(false))
        {
        }

        bool was_cancelled() const noexcept
        {
            return was_cancelled_->load(std::memory_order_relaxed);
        }

        void cancel() noexcept
        {
            was_cancelled_->store(true, std::memory_order_relaxed);
        }
    };
}}}    // namespace hpx::parallel::util
