//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/traits/is_invocable.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/futures/detail/future_data.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/lcos_local/promise.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace lcos { namespace local {
    template <typename Sig>
    class packaged_task;

    template <typename R, typename... Ts>
    class packaged_task<R(Ts...)>
    {
        typedef util::unique_function_nonser<R(Ts...)> function_type;

    public:
        // construction and destruction
        packaged_task()
          : function_()
          , promise_()
        {
        }

        template <typename F, typename FD = typename std::decay<F>::type,
            typename Enable = typename std::enable_if<
                !std::is_same<FD, packaged_task>::value &&
                traits::is_invocable_r<R, FD&, Ts...>::value>::type>
        explicit packaged_task(F&& f)
          : function_(std::forward<F>(f))
          , promise_()
        {
        }

        template <typename Allocator, typename F,
            typename FD = typename std::decay<F>::type,
            typename Enable = typename std::enable_if<
                !std::is_same<FD, packaged_task>::value &&
                traits::is_invocable_r<R, FD&, Ts...>::value>::type>
        explicit packaged_task(std::allocator_arg_t, Allocator const& a, F&& f)
          : function_(std::forward<F>(f))
          , promise_(std::allocator_arg, a)
        {
        }

        packaged_task(packaged_task&& rhs)
          : function_(std::move(rhs.function_))
          , promise_(std::move(rhs.promise_))
        {
        }

        packaged_task& operator=(packaged_task&& rhs)
        {
            if (this != &rhs)
            {
                function_ = std::move(rhs.function_);
                promise_ = std::move(rhs.promise_);
            }
            return *this;
        }

        void swap(packaged_task& rhs) noexcept
        {
            function_.swap(rhs.function_);
            promise_.swap(rhs.promise_);
        }

        void operator()(Ts... vs)
        {
            if (function_.empty())
            {
                HPX_THROW_EXCEPTION(no_state,
                    "packaged_task_base<Signature>::get_future",
                    "this packaged_task has no valid shared state");
                return;
            }

            hpx::util::annotate_function annotate(function_);
            invoke_impl(std::is_void<R>(), std::forward<Ts>(vs)...);
        }

        // result retrieval
        lcos::future<R> get_future(error_code& ec = throws)
        {
            if (function_.empty())
            {
                HPX_THROWS_IF(ec, no_state,
                    "packaged_task_base<Signature>::get_future",
                    "this packaged_task has no valid shared state");
                return lcos::future<R>();
            }
            return promise_.get_future();
        }

        bool valid() const noexcept
        {
            return !function_.empty() && promise_.valid();
        }

        void reset(error_code& ec = throws)
        {
            if (function_.empty())
            {
                HPX_THROWS_IF(ec, no_state,
                    "packaged_task_base<Signature>::get_future",
                    "this packaged_task has no valid shared state");
                return;
            }
            promise_ = local::promise<R>();
        }

        // extension
        void set_exception(std::exception_ptr const& e)
        {
            promise_.set_exception(e);
        }

    private:
        // synchronous execution
        template <typename... Vs>
        void invoke_impl(/*is_void=*/std::false_type, Vs&&... vs)
        {
            std::exception_ptr p;

            try
            {
                promise_.set_value(function_(std::forward<Vs>(vs)...));
                return;
            }
            catch (...)
            {
                p = std::current_exception();
            }

            // The exception is set outside the catch block since
            // set_exception may yield. Ending the catch block on a
            // different worker thread than where it was started may lead
            // to segfaults.
            promise_.set_exception(std::move(p));
        }

        template <typename... Vs>
        void invoke_impl(/*is_void=*/std::true_type, Vs&&... vs)
        {
            std::exception_ptr p;

            try
            {
                function_(std::forward<Ts>(vs)...);
                promise_.set_value();
                return;
            }
            catch (...)
            {
                p = std::current_exception();
            }

            // The exception is set outside the catch block since
            // set_exception may yield. Ending the catch block on a
            // different worker thread than where it was started may lead
            // to segfaults.
            promise_.set_exception(std::move(p));
        }

    private:
        function_type function_;
        local::promise<R> promise_;
    };
}}}    // namespace hpx::lcos::local

namespace std {
    // Requires: Allocator shall be an allocator (17.6.3.5)
    template <typename Sig, typename Allocator>
    struct uses_allocator<hpx::lcos::local::packaged_task<Sig>, Allocator>
      : std::true_type
    {
    };
}    // namespace std
