//  Copyright (c) 2022 Gregor Daiß
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_sycl/sycl_future.hpp>
#include <hpx/errors/exception.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/future_access.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace sycl { namespace experimental {

    struct sycl_executor
    {
        using future_type = hpx::future<void>;

        // -------------------------------------------------------------------------
        /// Create a SYCL executor (based on a sycl queue)
        sycl_executor(cl::sycl::default_selector selector)
          : command_queue(selector, cl::sycl::property::queue::in_order{})
        {
        }

        // TODO Future work: Add more constructors as needed (in case we need different queue constructors)
        // -------------------------------------------------------------------------
        // Queue will be cleaned up by its own destructor
        ~sycl_executor() {}

        /// Get future for this command_queue (NOTE will be more efficient if
        /// an event is provided -- otherwise a dummy kernel must be submitted to
        /// get an event)
        future_type get_future()
        {
            // The SYCL standard does not include a eventRecord method Instead
            // we have to submit some dummy function and use the event the
            // launch returns
            cl::sycl::event event = command_queue.submit(
                [&](cl::sycl::handler& h) { h.single_task([]() {}); });
            return detail::get_future(event);
        }

        /// Get future for that becomes ready when the given event completes
        inline future_type get_future(cl::sycl::event event)
        {
            return detail::get_future(event);
        }

#if defined(__INTEL_LLVM_COMPILER)
        // To find the correct overload (or any at all actually) we need to add the
        // code_location argument which is the last argument in every queue member
        // function in the intel oneapi sycl implementation.  As far as I can tell
        // it is usually invisible from the user-side since it is using a default
        // argument (code_location::current())

        /// sycl::queue::member_function type with code_location parameter
        template <typename... Params>
        using queue_function_ptr_t = cl::sycl::event (cl::sycl::queue::*)(
            Params..., const cl::sycl::detail::code_location&);
        /// Invoke member function given queue and parameters. Default
        /// code_location argument added automatically.
        template <typename... Params>
        void post(queue_function_ptr_t<Params...>&& queue_member_function,
            Params&&... args)
        {
            // for the intel version we need to actually pass the code location.
            // Within the intel sycl api this is usually a default argument,
            // but for invoke we need to pass it manually 
            cl::sycl::event e =
                std::invoke(std::forward<queue_function_ptr_t<Params...>>(
                                queue_member_function),
                    command_queue, std::forward<Params...>(args...),
                    cl::sycl::detail::code_location::current());
        }
        /// Invoke queue member function given queue and parameters. Default
        /// code_location argument added automatically.  / Returns hpx::future
        /// tied to the sycl event returned by the asynchronous queue member
        /// function call (two way)
        template <typename... Params>
        hpx::future<void> async_execute(
            queue_function_ptr_t<Params...>&& queue_member_function,
            Params&&... args)
        {
            // launching a sycl member function may throw -- if it does put it
            // into the future
            return hpx::detail::try_catch_exception_ptr(
                [&]() {
                    cl::sycl::event e = std::invoke(
                        std::forward<queue_function_ptr_t<Params...>>(
                            queue_member_function),
                        command_queue, std::forward<Params...>(args...),
                        cl::sycl::detail::code_location::current());
                    return get_future(e);
                },
                [&](std::exception_ptr&& ep) {
                    return hpx::make_exceptional_future<void>(HPX_MOVE(ep));
                });
        }
#else 
        // Default Implementation without the extra intel code_location parameter

        /// sycl::queue::member_function type 
        template<typename ...Params>
        using queue_function_ptr_t =  cl::sycl::event (cl::sycl::queue::*)(Params...);
        /// Invoke queue member function given queue and parameters -- do not
        /// use event to return a hpx::future (One way)
        template <typename... Params>
        void post(queue_function_ptr_t<Params...>&& queue_member_function,
            Params&&... args)
        {
            cl::sycl::event e =
                std::invoke(std::forward<queue_function_ptr_t<Params...>>(
                                queue_member_function),
                    command_queue, std::forward<Params...>(args...));
        }
        /// Invoke queue member function given queue and parameters --
        /// hpx::future tied to the sycl event / (two way)
        template <typename... Params>
        hpx::future<void> async_execute(
            queue_function_ptr_t<Params...>&& queue_member_function,
            Params&&... args)
        {
            // launching a sycl member function may throw -- if it does put it
            // into the future
            return hpx::detail::try_catch_exception_ptr(
                [&]() {
                    cl::sycl::event e = std::invoke(
                        std::forward<queue_function_ptr_t<Params...>>(
                            queue_member_function),
                        command_queue, std::forward<Params...>(args...));
                    return get_future(e);
                },
                [&](std::exception_ptr&& ep) {
                    return hpx::make_exceptional_future<void>(HPX_MOVE(ep));
                });
        }
#endif

        // Property interface:
        
        /// Return the device used by the underlying SYCL queue
        cl::sycl::device get_device() const {
          return command_queue.get_device();
        }
        // TODO Future work: Check if we want to expose any other (non-event) queue methods
      protected:
        cl::sycl::queue command_queue;

    };

}}}    // namespace hpx::sycl::experimental

namespace hpx { namespace parallel { namespace execution {

    /// \cond NOINTERNAL
    template <>
    struct is_one_way_executor<hpx::sycl::experimental::sycl_executor>
      : std::true_type
    {
        // support for fire and forget without returning a waitable/future
    };

    template <>
    struct is_two_way_executor<hpx::sycl::experimental::sycl_executor>
      : std::true_type
    {
        // support for a waitable/future
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution
