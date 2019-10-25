//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/ref.hpp
// hpxinspect:nodeprecatedname:boost::reference_wrapper

#include <hpx/config.hpp>

// #if defined(HPX_COMPUTE_DEVICE_CODE)
//
// #error "hpx::dataflow is not supported in device code"
//
// #else

// Intentionally #include future.hpp outside of the guards as it may #include
// dataflow.hpp itself
#include <hpx/lcos/future.hpp>

#ifndef HPX_LCOS_DATAFLOW_HPP
#define HPX_LCOS_DATAFLOW_HPP

#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/coroutines/detail/get_stack_pointer.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/functional/traits/is_action.hpp>
#include <hpx/lcos/detail/future_transforms.hpp>
#include <hpx/memory/intrusive_ptr.hpp>
#include <hpx/runtime/get_worker_thread_num.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/traits/acquire_future.hpp>
#include <hpx/traits/extract_action.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/traits/promise_local_result.hpp>
#include <hpx/type_support/always_void.hpp>
#include <hpx/util/annotated_function.hpp>
#include <hpx/util/pack_traversal_async.hpp>
#include <hpx/util/thread_description.hpp>

#include <hpx/parallel/executors/execution.hpp>
#include <hpx/parallel/executors/parallel_executor.hpp>

#include <boost/ref.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace detail
{
    template <typename F, typename Args>
    struct dataflow_not_callable
    {
#if defined(HPX_HAVE_CXX14_RETURN_TYPE_DEDUCTION)
        static auto error(F f, Args args)
        {
            hpx::util::invoke_fused(std::move(f), std::move(args));
        }
#else
        static auto error(F f, Args args)
         -> decltype(hpx::util::invoke_fused(std::move(f), std::move(args)));
#endif

        using type = decltype(
            error(std::declval<F>(), std::declval<Args>()));
    };

    ///////////////////////////////////////////////////////////////////////
    template <bool IsAction, typename F, typename Args, typename Enable = void>
    struct dataflow_return_impl
    {
        typedef typename dataflow_not_callable<F, Args>::type type;
    };

    template <typename Action, typename Args>
    struct dataflow_return_impl</*IsAction=*/true, Action, Args>
    {
        typedef typename Action::result_type type;
    };

    template <typename F, typename Args>
    struct dataflow_return_impl<
        /*IsAction=*/false, F, Args,
        typename hpx::util::always_void<
            typename hpx::util::detail::invoke_fused_result<F, Args>::type
        >::type
    > : util::detail::invoke_fused_result<F, Args>
    {};

    template <typename F, typename Args>
    struct dataflow_return
      : detail::dataflow_return_impl<traits::is_action<F>::value, F, Args>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Policy, typename Func, typename Futures>
    struct dataflow_frame //-V690
      : hpx::lcos::detail::future_data<
            typename detail::dataflow_return<Func, Futures>::type>
    {
        typedef
            typename detail::dataflow_return<Func, Futures>::type
            result_type;
        typedef hpx::lcos::detail::future_data<result_type> base_type;

        typedef hpx::lcos::future<result_type> type;

        typedef std::is_void<result_type> is_void;

    private:
        // workaround gcc regression wrongly instantiating constructors
        dataflow_frame();
        dataflow_frame(dataflow_frame const&);

    public:
        typedef typename base_type::init_no_addref init_no_addref;

        /// A struct to construct the dataflow_frame in-place
        struct construction_data
        {
            Policy policy_;
            Func func_;
        };

        /// Construct the dataflow_frame from the given policy
        /// and callable object.
        static construction_data construct_from(Policy policy, Func func)
        {
            return construction_data{std::move(policy), std::move(func)};
        }

        explicit dataflow_frame(construction_data data)
          : base_type(init_no_addref{})
          , policy_(std::move(data.policy_))
          , func_(std::move(data.func_))
        {
        }

    private:
        ///////////////////////////////////////////////////////////////////////
        /// Passes the futures into the evaluation function and
        /// sets the result future.
        HPX_FORCEINLINE
        void execute(std::false_type, Futures&& futures)
        {
            try {
                Func func = std::move(func_);

                this->set_data(
                    util::invoke_fused(std::move(func), std::move(futures)));
            }
            catch(...) {
                this->set_exception(std::current_exception());
            }
        }

        /// Passes the futures into the evaluation function and
        /// sets the result future.
        HPX_FORCEINLINE
        void execute(std::true_type, Futures&& futures)
        {
            try {
                Func func = std::move(func_);

                util::invoke_fused(std::move(func), std::move(futures));

                this->set_data(util::unused_type());
            }
            catch(...) {
                this->set_exception(std::current_exception());
            }
        }

        HPX_FORCEINLINE void done(Futures futures)
        {
            hpx::util::annotate_function annotate(func_);

            execute(is_void{}, std::move(futures));
        }

        ///////////////////////////////////////////////////////////////////////
        void finalize(hpx::detail::async_policy policy, Futures&& futures)
        {
            // schedule the final function invocation with high priority
            hpx::intrusive_ptr<dataflow_frame> this_(this);

            // simply schedule new thread
            parallel::execution::parallel_policy_executor<launch::async_policy>
                exec{policy};
            parallel::execution::post(exec,
                [HPX_CAPTURE_MOVE(this_)](Futures&& futures) -> void {
                    return this_->done(std::move(futures));
                }, std::move(futures));
        }

        HPX_FORCEINLINE
        void finalize(hpx::detail::sync_policy, Futures&& futures)
        {
            // We need to run the completion on a new thread if we are on a
            // non HPX thread.
            bool recurse_asynchronously = hpx::threads::get_self_ptr() == nullptr;
#if defined(HPX_HAVE_THREADS_GET_STACK_POINTER)
            recurse_asynchronously =
                !this_thread::has_sufficient_stack_space();
#else
            struct handle_continuation_recursion_count
            {
                handle_continuation_recursion_count()
                  : count_(threads::get_continuation_recursion_count())
                {
                    ++count_;
                }
                ~handle_continuation_recursion_count()
                {
                    --count_;
                }

                std::size_t& count_;
            } cnt;
            recurse_asynchronously = recurse_asynchronously ||
                cnt.count_ > HPX_CONTINUATION_MAX_RECURSION_DEPTH;
#endif
            if (!recurse_asynchronously)
            {
                done(std::move(futures));
            }
            else
            {
                finalize(hpx::launch::async, std::move(futures));
            }
        }

        void finalize(hpx::detail::fork_policy policy, Futures&& futures)
        {
            // schedule the final function invocation with high priority
            hpx::intrusive_ptr<dataflow_frame> this_(this);

            parallel::execution::parallel_policy_executor<launch::fork_policy>
                exec{policy};
            parallel::execution::post(exec,
                [HPX_CAPTURE_MOVE(this_)](Futures&& futures) -> void {
                    return this_->done(std::move(futures));
                }, std::move(futures));
        }

        void finalize(launch policy, Futures&& futures)
        {
            if (policy == launch::sync)
            {
                finalize(launch::sync, std::move(futures));
            }
            else if (policy == launch::fork)
            {
                finalize(launch::fork, std::move(futures));
            }
            else
            {
                finalize(launch::async, std::move(futures));
            }
        }

        // The overload for hpx::dataflow taking an executor simply forwards
        // to the corresponding executor customization point.
        //
        // parallel::execution::executor
        // threads::executor
        template <typename Executor>
        HPX_FORCEINLINE
        typename std::enable_if<
            traits::is_one_way_executor<Executor>::value ||
            traits::is_two_way_executor<Executor>::value ||
            traits::is_threads_executor<Executor>::value
        >::type
        finalize(Executor&& exec, Futures&& futures)
        {
            hpx::intrusive_ptr<dataflow_frame> this_(this);
            parallel::execution::post(std::forward<Executor>(exec),
                [HPX_CAPTURE_MOVE(this_)](Futures&& futures) -> void {
                    return this_->execute(is_void{}, std::move(futures));
                }, std::move(futures));
        }

    public:
        /// Check whether the current future is ready
        template <typename T>
        auto operator()(util::async_traverse_visit_tag, T&& current)
            -> decltype(async_visit_future(std::forward<T>(current)))
        {
            return async_visit_future(std::forward<T>(current));
        }

        /// Detach the current execution context and continue when the
        /// current future was set to be ready.
        template <typename T, typename N>
        auto operator()(util::async_traverse_detach_tag, T&& current, N&& next)
            -> decltype(async_detach_future(
                std::forward<T>(current), std::forward<N>(next)))
        {
            return async_detach_future(
                std::forward<T>(current), std::forward<N>(next));
        }

        /// Finish the dataflow when the traversal has finished
        HPX_FORCEINLINE void operator()(
            util::async_traverse_complete_tag, Futures futures)
        {
            finalize(policy_, std::move(futures));
        }

    private:
        Policy policy_;
        Func func_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Policy, typename Func, typename ...Ts,
        typename Frame = dataflow_frame<
            typename std::decay<Policy>::type,
            typename std::decay<Func>::type,
            util::tuple<typename std::decay<Ts>::type...>>>
    typename Frame::type create_dataflow(
        Policy && policy, Func && func, Ts &&... ts)
    {
        // Create the data which is used to construct the dataflow_frame
        auto data = Frame::construct_from(
            std::forward<Policy>(policy), std::forward<Func>(func));

        // Construct the dataflow_frame and traverse
        // the arguments asynchronously
        hpx::intrusive_ptr<Frame> p = util::traverse_pack_async(
            util::async_traverse_in_place_tag<Frame>{},
            std::move(data), std::forward<Ts>(ts)...);

        using traits::future_access;
        return future_access<typename Frame::type>::create(std::move(p));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Allocator, typename Policy, typename Func, typename ...Ts,
        typename Frame = dataflow_frame<
            typename std::decay<Policy>::type,
            typename std::decay<Func>::type,
            util::tuple<typename std::decay<Ts>::type...>>>
    typename Frame::type create_dataflow_alloc(
        Allocator const& alloc, Policy && policy, Func && func, Ts &&... ts)
    {
        // Create the data which is used to construct the dataflow_frame
        auto data = Frame::construct_from(
            std::forward<Policy>(policy), std::forward<Func>(func));

        // Construct the dataflow_frame and traverse
        // the arguments asynchronously
        hpx::intrusive_ptr<Frame> p = util::traverse_pack_async_allocator(
            alloc, util::async_traverse_in_place_tag<Frame>{},
            std::move(data), std::forward<Ts>(ts)...);

        using traits::future_access;
        return future_access<typename Frame::type>::create(std::move(p));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename FD, typename Enable = void>
    struct dataflow_dispatch;

    // launch
    template <typename Policy>
    struct dataflow_dispatch<Policy, typename std::enable_if<
            traits::is_launch_policy<Policy>::value
        >::type>
    {
        template <
            typename Allocator, typename Policy_,
            typename Component, typename Signature, typename Derived,
            typename ...Ts>
        HPX_FORCEINLINE static lcos::future<
            typename traits::promise_local_result<
                typename hpx::actions::basic_action<
                    Component, Signature, Derived>::remote_result_type
            >::type>
        call(Allocator const& alloc, Policy_ && policy,
            hpx::actions::basic_action<Component, Signature, Derived> const& act,
            naming::id_type const& id, Ts &&... ts)
        {
            return detail::create_dataflow_alloc(alloc,
                std::forward<Policy_>(policy), Derived{},
                id, traits::acquire_future_disp()(std::forward<Ts>(ts))...);
        }

        template <typename Allocator, typename Policy_, typename F,
            typename ...Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            !traits::is_action<typename std::decay<F>::type>::value,
            lcos::future<
                typename detail::dataflow_return<
                    typename std::decay<F>::type,
                    util::tuple<typename traits::acquire_future<Ts>::type...>
                >::type>
        >::type
        call(Allocator const& alloc, Policy_ && policy, F && f,
            Ts &&... ts)
        {
            return detail::create_dataflow_alloc(alloc,
                std::forward<Policy_>(policy), std::forward<F>(f),
                traits::acquire_future_disp()(std::forward<Ts>(ts))...);
        }
    };

    // parallel executors
    // threads::executor
    template <typename Executor>
    struct dataflow_dispatch<Executor, typename std::enable_if<
            traits::is_one_way_executor<Executor>::value ||
            traits::is_two_way_executor<Executor>::value ||
            traits::is_threads_executor<Executor>::value
        >::type>
    {
        template <typename Allocator, typename Executor_, typename F,
            typename ...Ts>
        HPX_FORCEINLINE static typename std::enable_if<
            !traits::is_action<typename std::decay<F>::type>::value,
            lcos::future<
                typename detail::dataflow_return<
                    typename std::decay<F>::type,
                    util::tuple<typename traits::acquire_future<Ts>::type...>
                >::type>
        >::type
        call(Allocator const& alloc, Executor_ && exec, F && f, Ts &&... ts)
        {
            return detail::create_dataflow_alloc(alloc,
                std::forward<Executor_>(exec), std::forward<F>(f),
                traits::acquire_future_disp()(std::forward<Ts>(ts))...);
        }
    };

    // any action, plain function, or function object
    template <typename FD>
    struct dataflow_dispatch<FD, typename std::enable_if<
        !traits::is_launch_policy<FD>::value &&
        !(
            traits::is_one_way_executor<FD>::value ||
            traits::is_two_way_executor<FD>::value ||
            traits::is_threads_executor<FD>::value)
        >::type>
    {
        template <
            typename Allocator, typename Component, typename Signature,
            typename Derived, typename ...Ts>
        HPX_FORCEINLINE static auto
        call(Allocator const& alloc,
            hpx::actions::basic_action<Component, Signature, Derived> const& act,
            naming::id_type const& id, Ts &&... ts)
        ->  decltype(dataflow_dispatch<launch>::call(
                alloc, launch::async, act, id, std::forward<Ts>(ts)...))
        {
            return dataflow_dispatch<launch>::call(
                alloc, launch::async, act, id, std::forward<Ts>(ts)...);
        }

        template <
            typename Allocator, typename F, typename ...Ts,
            typename Enable = typename std::enable_if<
                !traits::is_action<typename std::decay<F>::type>::value
            >::type>
        HPX_FORCEINLINE static auto
        call(Allocator const& alloc, F && f, Ts &&... ts)
        ->  decltype(dataflow_dispatch<launch>::call(
                alloc, launch::async, std::forward<F>(f),
                std::forward<Ts>(ts)...))
        {
            return dataflow_dispatch<launch>::call(
                alloc, launch::async, std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename T0, typename Enable = void>
    struct dataflow_action_dispatch
    {
        template <typename Allocator, typename ...Ts>
        HPX_FORCEINLINE static lcos::future<
            typename traits::promise_local_result<
                typename hpx::traits::extract_action<Action>::remote_result_type
            >::type>
        call(Allocator const& alloc, naming::id_type const& id,
            Ts &&... ts)
        {
            return dataflow_dispatch<Action>::call(alloc,
                Action(), id, std::forward<Ts>(ts)...);
        }
    };

    template <typename Action, typename Policy>
    struct dataflow_action_dispatch<Action, Policy, typename std::enable_if<
            traits::is_launch_policy<typename std::decay<Policy>::type>::value
        >::type>
    {
        template <typename Allocator, typename ...Ts>
        HPX_FORCEINLINE static lcos::future<
            typename traits::promise_local_result<
                typename hpx::traits::extract_action<Action>::remote_result_type
            >::type>
        call(Allocator const& alloc, Policy && policy,
            naming::id_type const& id, Ts &&... ts)
        {
            return dataflow_dispatch<typename std::decay<Policy>::type>::
                call(alloc, std::forward<Policy>(policy), Action(), id,
                    std::forward<Ts>(ts)...);
        }
    };
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    template <typename F, typename ...Ts>
    HPX_FORCEINLINE
    auto dataflow(F && f, Ts &&... ts)
    ->  decltype(
            lcos::detail::dataflow_dispatch<typename std::decay<F>::type>::call(
                hpx::util::internal_allocator<>{}, std::forward<F>(f),
                std::forward<Ts>(ts)...
        ))
    {
        return lcos::detail::dataflow_dispatch<typename std::decay<F>::type>::
            call(hpx::util::internal_allocator<>{}, std::forward<F>(f),
                std::forward<Ts>(ts)...);
    }

    template <typename Allocator, typename F, typename ...Ts>
    HPX_FORCEINLINE
    auto dataflow_alloc(Allocator const& alloc, F && f, Ts &&... ts)
    ->  decltype(
            lcos::detail::dataflow_dispatch<typename std::decay<F>::type>::
                call(alloc, std::forward<F>(f), std::forward<Ts>(ts)...
        ))
    {
        return lcos::detail::dataflow_dispatch<typename std::decay<F>::type>::
            call(alloc, std::forward<F>(f), std::forward<Ts>(ts)...);
    }

    template <
        typename Action, typename T0, typename ...Ts,
        typename Enable = typename std::enable_if<
            traits::is_action<Action>::value>::type>
    HPX_FORCEINLINE
    auto dataflow(T0 && t0, Ts &&... ts)
    ->  decltype(lcos::detail::dataflow_action_dispatch<Action, T0>::call(
            hpx::util::internal_allocator<>{}, std::forward<T0>(t0),
            std::forward<Ts>(ts)...))
    {
        return lcos::detail::dataflow_action_dispatch<Action, T0>::call(
            hpx::util::internal_allocator<>{}, std::forward<T0>(t0),
            std::forward<Ts>(ts)...);
    }

    template <
        typename Action, typename Allocator, typename T0, typename ...Ts,
        typename Enable = typename std::enable_if<
            traits::is_action<Action>::value>::type>
    HPX_FORCEINLINE
    auto dataflow_alloc(Allocator const& alloc, T0 && t0, Ts &&... ts)
    ->  decltype(lcos::detail::dataflow_action_dispatch<Action, T0>::call(
            alloc, std::forward<T0>(t0), std::forward<Ts>(ts)...))
    {
        return lcos::detail::dataflow_action_dispatch<Action, T0>::call(
            alloc, std::forward<T0>(t0), std::forward<Ts>(ts)...);
    }
}

// #endif

#endif /*HPX_LCOS_DATAFLOW_HPP*/
