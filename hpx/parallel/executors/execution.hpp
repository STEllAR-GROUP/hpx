//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nounnamed

/// \file parallel/executors/execution.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_EXECUTION_DEC_23_0712PM)
#define HPX_PARALLEL_EXECUTORS_EXECUTION_DEC_23_0712PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/traits/detail/wrap_int.hpp>
#include <hpx/traits/future_then_result.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/detected.hpp>
#include <hpx/util/unwrapped.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/execution_fwd.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(concurrency_v2) {
    namespace execution
{
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct executor_context
    {
        using type =
            typename std::decay<
                decltype(std::declval<Executor const&>().context())
            >::type;
    };

    template <typename Executor>
    using executor_context_t = typename executor_context<Executor>::type;

    ///////////////////////////////////////////////////////////////////////////
    // Components which create groups of execution agents may use execution
    // categories to communicate the forward progress and ordering guarantees
    // of these execution agents with respect to other agents within the same
    // group.

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct executor_execution_category
    {
    private:
        template <typename T>
        using execution_category = typename T::execution_category;

    public:
        using type = hpx::util::detected_or_t<
            unsequenced_execution_tag, execution_category, Executor>;
    };

    template <typename Executor>
    using executor_execution_category_t =
        typename executor_execution_category<Executor>::type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct executor_shape
    {
    private:
        template <typename T>
        using shape_type = typename T::shape_type;

    public:
        using type = hpx::util::detected_or_t<
            std::size_t, shape_type, Executor>;
    };

    template <typename Executor>
    using executor_shape_t = typename executor_shape<Executor>::type;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct executor_index
    {
    private:
        // exposition only
        template <typename T>
        using index_type = typename T::index_type;

    public:
        using type = hpx::util::detected_or_t<
            executor_shape_t<Executor>, index_type, Executor>;
    };

    template <typename Executor>
    using executor_index_t = typename executor_index<Executor>::type;

    ///////////////////////////////////////////////////////////////////////////
    // Executor customization points
    namespace detail
    {
        template <typename Executor, typename Enable = void>
        struct execute_fn_helper;

        template <typename Executor, typename Enable = void>
        struct async_execute_fn_helper;

        template <typename Executor, typename Enable = void>
        struct sync_execute_fn_helper;

        template <typename Executor, typename Enable = void>
        struct then_execute_fn_helper;

        template <typename Executor, typename Enable = void>
        struct post_fn_helper;
    }

    // customization point for OneWayExecutor interface
    // execute()
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // default implementation of the execute() customization point
        template <typename OneWayExecutor, typename F, typename ... Ts>
        HPX_FORCEINLINE auto execute(OneWayExecutor const& exec, F && f,
                Ts &&... ts)
        ->  decltype(exec.execute(std::forward<F>(f), std::forward<Ts>(ts)...))
        {
            exec.execute(std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename Executor>
        struct execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value
            >::type>
        {
            template <typename OneWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static auto call(OneWayExecutor && exec, F && f,
                    Ts &&... ts)
            ->  decltype(
                    execute(exec, std::forward<F>(f), std::forward<Ts>(ts)...)
                )
            {
                return execute(exec, std::forward<F>(f),
                    std::forward<Ts>(ts)...);
            }
        };

        struct execute_fn
        {
            template <typename Executor, typename F, typename ... Ts>
            HPX_FORCEINLINE auto operator()(Executor const& exec,
                    F && f, Ts &&... ts) const
            ->  decltype(execute_fn_helper<Executor>::call(
                    exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ))
            {
                return execute_fn_helper<Executor>::call(exec,
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // emulate async_execute() on OneWayExecutors
        template <typename Executor>
        struct async_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value &&
               !hpx::traits::is_two_way_executor<Executor>::value
            >::type>
        {
            template <typename OneWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static auto call(OneWayExecutor const& exec,
                    F && f, Ts &&... ts)
            ->  hpx::future<decltype(execute(
                    exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ))>
            {
                return hpx::lcos::make_ready_future(execute(
                        exec, std::forward<F>(f), std::forward<Ts>(ts)...
                    ));
            }
        };

        // emulate sync_execute() on OneWayExecutors
        template <typename Executor>
        struct sync_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value &&
               !hpx::traits::is_two_way_executor<Executor>::value
            >::type>
        {
            template <typename OneWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static auto call(OneWayExecutor const& exec,
                    F && f, Ts &&... ts)
            ->  decltype(execute(
                    exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ))
            {
                return execute(exec, std::forward<F>(f),
                    std::forward<Ts>(ts)...);
            }
        };

        // emulate then_execute() on OneWayExecutors
        template <typename Executor>
        struct then_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value &&
               !hpx::traits::is_two_way_executor<Executor>::value
            >::type>
        {
            template <typename OneWayExecutor, typename F, typename Future,
                typename ... Ts>
            HPX_FORCEINLINE static
            typename hpx::traits::future_then_result<F, Ts...>::type
            call(OneWayExecutor const& exec, F && f, Future& predecessor,
                Ts &&... ts)
            {
                return predecessor.then(hpx::util::bind(
                        hpx::util::one_shot(std::forward<F>(f)),
                        hpx::util::placeholders::_1, std::forward<Ts>(ts)...
                    ));
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // customization points for TwoWayExecutor interface
    // async_execute(), sync_execute(), then_execute()
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // default implementation of the async_execute() customization point
        template <typename TwoWayExecutor, typename F, typename ... Ts>
        HPX_FORCEINLINE auto async_execute(TwoWayExecutor const& exec, F && f,
                Ts &&... ts)
        ->  decltype(exec.async_execute(
                std::forward<F>(f), std::forward<Ts>(ts)...
            ))
        {
            return exec.async_execute(std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        template <typename Executor>
        struct async_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_two_way_executor<Executor>::value
            >::type>
        {
            template <typename TwoWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static auto call(TwoWayExecutor const& exec,
                    F && f, Ts &&... ts)
            ->  decltype(async_execute(
                    exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ))
            {
                return async_execute(exec, std::forward<F>(f),
                    std::forward<Ts>(ts)...);
            }
        };

        struct async_execute_fn
        {
            template <typename Executor, typename F, typename ... Ts>
            auto operator()(Executor const& exec, F && f, Ts &&... ts) const
            ->  decltype(async_execute_fn_helper<Executor>::call(
                    exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ))
            {
                return async_execute_fn_helper<Executor>::call(exec,
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // default implementation of the sync_execute() customization point
        template <typename TwoWayExecutor, typename F, typename ... Ts>
        HPX_FORCEINLINE auto sync_execute(TwoWayExecutor const& exec,
                F && f, Ts &&... ts)
        ->  decltype(
                exec.sync_execute(std::forward<F>(f), std::forward<Ts>(ts)...)
            )
        {
            return exec.sync_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename Executor>
        struct sync_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_two_way_executor<Executor>::value
            >::type>
        {
            template <typename TwoWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static auto call(TwoWayExecutor const& exec,
                    F && f, Ts &&... ts)
            ->  decltype(sync_execute(
                    exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ))
            {
                return sync_execute(exec, std::forward<F>(f),
                    std::forward<Ts>(ts)...);
            }
        };

        struct sync_execute_fn
        {
            template <typename Executor, typename F, typename ... Ts>
            HPX_FORCEINLINE auto operator()(Executor const& exec,
                    F && f, Ts &&... ts) const
            ->  decltype(sync_execute_fn_helper<Executor>::call(
                    exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ))
            {
                return sync_execute_fn_helper<Executor>::call(exec,
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // default implementation of the then_execute() customization point
        template <typename TwoWayExecutor, typename F, typename Future,
            typename ... Ts>
        HPX_FORCEINLINE auto then_execute(TwoWayExecutor const& exec, F && f,
                Future& predecessor, Ts &&... ts)
        ->  decltype(exec.then_execute(
                std::forward<F>(f), predecessor, std::forward<Ts>(ts)...
            ))
        {
            return exec.then_execute(std::forward<F>(f),
                predecessor, std::forward<Ts>(ts)...);
        }

        template <typename Executor>
        struct then_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_two_way_executor<Executor>::value
            >::type>
        {
            template <typename TwoWayExecutor, typename F, typename Future,
                typename ... Ts>
            HPX_FORCEINLINE static auto call(TwoWayExecutor const& exec,
                    F && f, Future& predecessor, Ts &&... ts)
            ->  decltype(then_execute(
                    exec, std::forward<F>(f), predecessor,
                    std::forward<Ts>(ts)...
                ))
            {
                return then_execute(exec, std::forward<F>(f),
                    predecessor, std::forward<Ts>(ts)...);
            }
        };

        struct then_execute_fn
        {
            template <typename Executor, typename F, typename Future,
                typename ... Ts>
            HPX_FORCEINLINE auto operator()(Executor const& exec,
                    F && f, Future& predecessor, Ts &&... ts) const
            ->  decltype(then_execute_fn_helper<Executor>::call(
                    exec, std::forward<F>(f), predecessor,
                    std::forward<Ts>(ts)...
                ))
            {
                return then_execute_fn_helper<Executor>::call(exec,
                    std::forward<F>(f), predecessor,
                    std::forward<Ts>(ts)...);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // emulate post() on TwoWayExecutors
        template <typename Executor>
        struct post_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_two_way_executor<Executor>::value &&
               !hpx::traits::is_non_blocking_one_way_executor<Executor>::value
            >::type>
        {
            // dispatch to V1 executors
            template <typename TwoWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static auto
            call_impl(int, TwoWayExecutor const& exec, F && f, Ts &&... ts)
            ->  decltype(exec.apply_execute(
                    std::forward<F>(f), std::forward<Ts>(ts)...
                ))
            {
                // use apply_execute, if exposed
                exec.apply_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static void
            call_impl(hpx::traits::detail::wrap_int, TwoWayExecutor const& exec,
                F && f, Ts &&... ts)
            {
                // simply discard the returned future
                async_execute(exec, std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static auto call(TwoWayExecutor const& exec,
                    F && f, Ts &&... ts)
            ->  decltype(call_impl(
                    0, exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ))
            {
                // simply discard the returned future
                return call_impl(0, exec, std::forward<F>(f),
                    std::forward<Ts>(ts)...);
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // post()
    namespace detail
    {
        // default implementation of the post() customization point
        template <typename NonBlockingOneWayExecutor, typename F,
            typename ... Ts>
        HPX_FORCEINLINE auto post(NonBlockingOneWayExecutor const& exec,
                F && f, Ts &&... ts)
        ->  decltype(
                exec.post(std::forward<F>(f), std::forward<Ts>(ts)...)
            )
        {
            return exec.post(std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename Executor>
        struct post_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_non_blocking_one_way_executor<Executor>::value
            >::type>
        {
            template <typename NonBlockingOneWayExecutor, typename F,
                typename ... Ts>
            HPX_FORCEINLINE static auto call(
                    NonBlockingOneWayExecutor const& exec, F && f, Ts &&... ts)
            ->  decltype(post(
                    exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ))
            {
                return post(exec, std::forward<F>(f), std::forward<Ts>(ts)...);
            }
        };

        struct post_fn
        {
            template <typename Executor, typename F, typename ... Ts>
            HPX_FORCEINLINE auto operator()(Executor const& exec, F && f,
                    Ts &&... ts) const
            ->  decltype(post_fn_helper<Executor>::call(
                    exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ))
            {
                return post_fn_helper<Executor>::call(exec,
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // Helper facility to avoid ODR violations
        template <typename T>
        struct static_const
        {
            HPX_STATIC_CONSTEXPR T value{};
        };
    }
    /// \endcond

    namespace
    {
        // OneWayExecutor customization point: execution::execute

        /// Customization point for synchronous execution agent creation.
        ///
        /// This synchronously creates a single function invocation f() using
        /// the associated executor. The execution of the supplied function
        /// synchronizes with the caller
        ///
        /// \param exec [in] The executor object to use for scheduling of the
        ///             function \a f.
        /// \param f    [in] The function which will be scheduled using the
        ///             given executor.
        /// \param ts   [in] Additional arguments to use to invoke \a f.
        ///
        /// \returns f(ts...)'s result
        ///
        /// \note This is valid for one way executors only, it will call
        ///       exec.execute(f, ts...) if it exists.
        ///
        constexpr detail::execute_fn const& execute =
            detail::static_const<detail::execute_fn>::value;

        // TwoWayExecutor customization points: execution::async_execute,
        // execution::sync_execute, and execution::then_execute

        /// Customization point for asynchronous execution agent creation.
        ///
        /// This asynchronously creates a single function invocation f() using
        /// the associated executor.
        ///
        /// \param exec [in] The executor object to use for scheduling of the
        ///             function \a f.
        /// \param f    [in] The function which will be scheduled using the
        ///             given executor.
        /// \param ts   [in] Additional arguments to use to invoke \a f.
        ///
        /// \note Executors have to implement only `async_execute()`. All other
        ///       functions will be emulated by this `executor_traits` in terms
        ///       of this single basic primitive. However, some executors will
        ///       naturally specialize all operations for maximum efficiency.
        ///
        /// \note This is valid for one way executors (calls
        ///       make_ready_future(exec.execute(f, ts...) if it exists) and
        ///       for two way executors (calls exec.async_execute(f, ts...) if
        ///       it exists).
        ///
        /// \returns f(ts...)'s result through a future
        ///
        constexpr detail::async_execute_fn const& async_execute =
            detail::static_const<detail::async_execute_fn>::value;

        /// Customization point for synchronous execution agent creation.
        ///
        /// This synchronously creates a single function invocation f() using
        /// the associated executor. The execution of the supplied function
        /// synchronizes with the caller
        ///
        /// \param exec [in] The executor object to use for scheduling of the
        ///             function \a f.
        /// \param f    [in] The function which will be scheduled using the
        ///             given executor.
        /// \param ts   [in] Additional arguments to use to invoke \a f.
        ///
        /// \returns f(ts...)'s result
        ///
        /// \note This is valid for two way executors (calls
        ///       exec.sync_execute(f, ts...) if it exists) and for one way
        ///       executors (calls exec.execute(f, ts...) if it exists).
        ///
        constexpr detail::sync_execute_fn const& sync_execute =
            detail::static_const<detail::sync_execute_fn>::value;

        /// Customization point for synchronous execution agent creation.
        ///
        /// This synchronously creates a single function invocation f() using
        /// the associated executor. The execution of the supplied function
        /// synchronizes with the caller
        ///
        /// \param exec [in] The executor object to use for scheduling of the
        ///             function \a f.
        /// \param f    [in] The function which will be scheduled using the
        ///             given executor.
        /// \param ts   [in] Additional arguments to use to invoke \a f.
        ///
        /// \returns f(ts...)'s result through a future
        ///
        /// \note This is valid for two way executors (calls
        ///       exec.then_execute(f, predecessor, ts...) if it exists) and
        ///       for one way executors (calls predecessor.then(bind(f, ts...))).
        ///
        constexpr detail::then_execute_fn const& then_execute =
            detail::static_const<detail::then_execute_fn>::value;

        // NonBlockingOneWayExecutor customization point: execution::post

        /// Customization point for asynchronous fire & forget execution
        /// agent creation.
        ///
        /// This asynchronously (fire & forget) creates a single function
        /// invocation f() using the associated executor.
        ///
        /// \param exec [in] The executor object to use for scheduling of the
        ///             function \a f.
        /// \param f    [in] The function which will be scheduled using the
        ///             given executor.
        /// \param ts   [in] Additional arguments to use to invoke \a f.
        ///
        /// \note This is valid for two way executors (calls
        ///       exec.apply_execute(f, ts...), if available, otherwise
        ///       it calls exec.async_execute(f, ts...) while discarding the
        ///       returned future), and for non-blocking two way executors
        ///       (calls exec.post(f, ts...) if it exists).
        ///
        constexpr detail::post_fn const& post =
            detail::static_const<detail::post_fn>::value;
    }

    ///////////////////////////////////////////////////////////////////////////

//   template<class TwoWayExecutor, class Function>
//     result_of_t<decay_t<Function>()>
//       sync_execute(const TwoWayExecutor& exec, Function&& f);
//   template<class OneWayExecutor, class Function>
//     result_of_t<decay_t<Function>()>
//       sync_execute(const OneWayExecutor& exec, Function&& f);
//
//   template<class TwoWayExecutor, class Function>
//     executor_future_t<TwoWayExecutor, result_of_t<decay_t<Function>()>>
//       async_execute(const TwoWayExecutor& exec, Function&& f);
//   template<class Executor, class Function>
//     std::future<decay_t<result_of_t<decay_t<Function>()>>
//       async_execute(const Executor& exec, Function&& f);
//
//   template<class NonBlockingTwoWayExecutor, class Function>
//     executor_future_t<NonBlockingTwoWayExecutor, result_of_t<decay_t<Function>()>>
//       async_post(const NonBlockingTwoWayExecutor& exec, Function&& f);
//   template<class NonBlockingOneWayExecutor, class Function>
//     std::future<decay_t<result_of_t<decay_t<Function>()>>
//       async_post(const NonBlockingOneWayExecutor& exec, Function&& f);
//
//   template<class NonBlockingTwoWayExecutor, class Function>
//     executor_future_t<NonBlockingTwoWayExecutor, result_of_t<decay_t<Function>()>>
//       async_defer(const NonBlockingTwoWayExecutor& exec, Function&& f);
//   template<class NonBlockingOneWayExecutor, class Function>
//     std::future<decay_t<result_of_t<decay_t<Function>()>>
//       async_defer(const NonBlockingOneWayExecutor& exec, Function&& f);
//
//   template<class TwoWayExecutor, class Function, class Future>
//     executor_future_t<TwoWayExecutor, see-below>
//       then_execute(const TwoWayExecutor& exec, Function&& f, Future& predecessor);
//   template<class OneWayExecutor, class Function, class Future>
//     executor_future_t<OneWayExecutor, see-below>
//       then_execute(const OneWayExecutor& exec, Function&& f, Future& predecessor);
//
//   template<class BulkOneWayExecutor, class Function1, class Function2>
//     void bulk_execute(const BulkOneWayExecutor& exec, Function1 f,
//                       executor_shape_t<BulkOneWayExecutor> shape,
//                       Function2 shared_factory);
//   template<class OneWayExecutor, class Function1, class Function2>
//     void bulk_execute(const OneWayExecutor& exec, Function1 f,
//                       executor_shape_t<OneWayExecutor> shape,
//                       Function2 shared_factory);
//
//   template<class BulkTwoWayExecutor, class Function1, class Function2, class Function3>
//     result_of_t<Function2()>
//       bulk_sync_execute(const BulkTwoWayExecutor& exec, Function1 f,
//                         executor_shape_t<BulkTwoWayExecutor> shape,
//                         Function2 result_factory, Function3 shared_factory);
//   template<class OneWayExecutor, class Function1, class Function2, class Function3>
//     result_of_t<Function2()>
//       bulk_sync_execute(const OneWayExecutor& exec, Function1 f,
//                         executor_shape_t<OneWayExecutor> shape,
//                         Function2 result_factory, Function3 shared_factory);
//
//   template<class BulkTwoWayExecutor, class Function1, class Function2, class Function3>
//     executor_future_t<const BulkTwoWayExecutor, result_of_t<Function2()>>
//       bulk_async_execute(const BulkTwoWayExecutor& exec, Function1 f,
//                          executor_shape_t<BulkTwoWayExecutor> shape,
//                          Function2 result_factory, Function3 shared_factory);
//   template<class OneWayExecutor, class Function1, class Function2, class Function3>
//     executor_future_t<const OneWayExecutor, result_of_t<Function2()>>
//       bulk_async_execute(const OneWayExecutor& exec, Function1 f,
//                          executor_shape_t<OneWayExecutor> shape,
//                          Function2 result_factory, Function3 shared_factory);
//
//   template<class BulkTwoWayExecutor, class Function1, class Future, class Function2, class Function3>
//     executor_future_t<Executor, result_of_t<Function2()>>
//       bulk_then_execute(const BulkTwoWayExecutor& exec, Function1 f,
//                         executor_shape_t<BulkTwoWayExecutor> shape,
//                         Future& predecessor,
//                         Function2 result_factory, Function3 shared_factory);
//   template<class OneWayExecutor, class Function1, class Future, class Function2, class Function3>
//     executor_future_t<OneWayExecutor, result_of_t<Function2()>>
//       bulk_then_execute(const OneWayExecutor& exec, Function1 f,
//                         executor_shape_t<OneWayExecutor> shape,
//                         Future& predecessor,
//                         Function2 result_factory, Function3 shared_factory);
//
//   // Executor work guard:
//
//   template <class Executor>
//     class executor_work_guard;
//
//   // Polymorphic executor wrappers:
//
//   class one_way_executor;
//   class host_based_one_way_executor;
//   class non_blocking_one_way_executor;
//   class two_way_executor;
//   class non_blocking_two_way_executor;
}}}}

#endif

