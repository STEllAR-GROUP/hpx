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

    namespace detail {
        // -------------------------------------------------------------------------
        // A helper object to call a cudafunction returning a cudaError type
        // or a plain kernel definition (or cublas function in cublas executor)
        template <typename R, typename... Args>
        struct dispatch_helper;

        // default implementation - call the function
        template <typename R, typename... Args>
        struct dispatch_helper
        {
            inline R operator()(R (*f)(Args...), Args... args)
            {
                return f(args...);
            }
        };

        // specialization for return type void
        template <typename... Args>
        struct dispatch_helper<void, Args...>
        {
            inline void operator()(void (*f)(Args...), Args... args)
            {
                f(args...);
            }
        };

        // specialization for return type of cudaError_t
        template <typename... Args>
        struct dispatch_helper<cl::sycl::event, Args...>
        {
            inline void operator()(cl::sycl::event (*f)(Args...), Args... args)
            {
                check_cuda_error(f(args...));
            }
        };

    }    // namespace detail

    // -------------------------------------------------------------------------
    // Allows the launching of cuda functions and kernels on a stream with futures
    // returned that are set when the async functions/kernels are ready
    // -------------------------------------------------------------------------
    struct sycl_executor
    {
        using future_type = hpx::future<void>;

        // -------------------------------------------------------------------------
        // constructors - create a cuda stream that all tasks invoked by
        // this helper will use
        // assume event mode is the default
        sycl_executor(cl::sycl::default_selector selector)
          : command_queue(selector, cl::sycl::property::queue::in_order{})
        {
        }

        // -------------------------------------------------------------------------
        // Queue will be cleaned up by its own destructor
        ~sycl_executor() {}

        /// Get future for this command_queue (NOTE will be more efficient if
        //an event is provided -- otherwise a dummy kernel must be submitted to
        //get an event)
        inline future_type get_future()
        {
            // The SYCL standard does not include a eventRecord method Instead
            // we have to submit some dummy function and use the event the
            // launch returns
            cl::sycl::event event = command_queue.submit(
                [&](cl::sycl::handler& h) { h.single_task([=]() {}); });
            return detail::get_future(command_queue, event);
        }

        /// Get future for that becomes ready when the given event completes
        inline future_type get_future(cl::sycl::event event)
        {
            return detail::get_future(command_queue, event);
        }

        // -------------------------------------------------------------------------
        // TODO(daissgr) Seems odd to accept more parameters here than in our async method...
        // OneWay Execution
        template <typename F, typename... Ts>
        inline decltype(auto) post(F&& f, Ts&&... ts)
        {
            return apply(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        // -------------------------------------------------------------------------
        // TwoWay Execution
        template <typename F, typename... Ts>
        inline decltype(auto) async_execute(F&& f, Ts&&... ts)
        {
            return async(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        /// For direct access to the underlying queue
        cl::sycl::queue& queue(void) {return command_queue;}

      protected:
        cl::sycl::queue command_queue;
        // -------------------------------------------------------------------------

        // Original try, did not work with dpcp++ due to different overloads on the device pass?
        // TODO(daissgr) Try again? 
        /* template <typename... Params, typename... Args> */
        /* void apply(cl::sycl::event (cl::sycl::queue::*queue_member_function)(Args...), */
        /*     Args&&... args) */
        /* { */
        /*     cl::sycl::event ret = (command_queue.*queue_member_function)( */
        /*         HPX_FORWARD(Args, args)...); */
        /* } */

        /// Just runs the given Functor using the queue of this executor
        void apply(std::function<cl::sycl::event(cl::sycl::queue&)> &&f)
        {
            // discard event in the one way case...
            cl::sycl::event e = f(command_queue);
        }

        /// Runs the given Functor, and uses its return event to create a future
        hpx::future<void> async(std::function<cl::sycl::event(cl::sycl::queue&)> &&f)
        {
            return hpx::detail::try_catch_exception_ptr(
                [&]() {
                    cl::sycl::event e = f(command_queue);
                    return get_future(e);
                },
                [&](std::exception_ptr&& ep) {
                    return hpx::make_exceptional_future<void>(HPX_MOVE(ep));
                });
        }
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
