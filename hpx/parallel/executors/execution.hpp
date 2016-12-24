//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/execution.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_EXECUTION_DEC_23_0712PM)
#define HPX_PARALLEL_EXECUTORS_EXECUTION_DEC_23_0712PM

#include <hpx/config.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/detected.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/execution_fwd.hpp>
#include <hpx/parallel/executors/rebind_executor.hpp>

#include <cstddef>
#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(concurrency_v2) {
    namespace execution
{
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
    namespace detail
    {
        template <typename Executor, typename F, typename Pack,
            typename Enable = void>
        struct executor_future;

        template <typename Executor, typename F, typename ... Ts>
        struct executor_future<Executor, F, hpx::util::detail::pack<Ts...>,
            typename std::enable_if<
                hpx::traits::is_two_way_executor<Executor>::value
            >::type>
        {
            using type = decltype(
                std::declval<Executor const&>().async_execute(
                    std::declval<F(*)(Ts...)>(), std::declval<Ts>()...));
        };

        template <typename Executor, typename T, typename Pack>
        struct executor_future<Executor, T, Pack,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value
            >::type>
        {
            using type = hpx::future<T>;
        };
    }

    template <typename Executor, typename T, typename ... Ts>
    struct executor_future
      : detail::executor_future<Executor, T, hpx::util::detail::pack<Ts...> >
    {};

    template <typename Executor, typename T, typename ... Ts>
    using executor_future_t = typename executor_future<Executor, T, Ts...>::type;

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

    // customization point for OneWayExecutor interface
    // execute()
    namespace detail
    {
        // default implementation of the execute() customization point
        template <typename OneWayExecutor, typename F, typename ... Ts>
        HPX_FORCEINLINE auto execute(OneWayExecutor const& exec, F && f,
                Ts &&... ts)
        ->  decltype(exec.execute(std::forward<F>(f), std::forward<Ts>(ts)...))
        {
            exec.execute(std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename Executor,
            typename std::enable_if<
                is_one_way_executor<Executor>::value
            >::type>
        struct execute_fn_helper
        {
            template <typename OneWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static auto call(OneWayExecutor && exec, F && f,
                    Ts &&... ts)
            ->  decltype(
                    execute(exec, std::forward<F>(f), std::forward<Ts>(ts)...)
                )
            {
                execute(exec, std::forward<F>(f), std::forward<Ts>(ts)...);
            }
        };

        template <typename Executor, typename F, typename ... Ts>
        HPX_FORCEINLINE auto execute_fn::operator()(Executor const& exec,
                F && f, Ts &&... ts) const
        ->  decltype(execute_fn_helper<Executor>::call(
                exec, std::forward<F>(f), std::forward<Ts>(ts)...
            ))
        {
            return execute_fn_helper<Executor>::call(exec,
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename Executor,
            typename std::enable_if<
               !is_two_way_executor<Executor>::value &&
                is_one_way_executor<Executor>::value
            >::type>
        struct async_execute_fn_helper
        {
            template <typename TwoWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static auto call(TwoWayExecutor const& exec,
                    F && f, Ts &&... ts)
            ->  hpx::future<decltype(execute(
                    exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ))>
            {
                return hpx::make_ready_future(execute(
                        exec, std::forward<F>(f), std::forward<Ts>(ts)...
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

        template <typename Executor,
            typename std::enable_if<
                is_two_way_executor<Executor>::value
            >::type>
        struct async_execute_fn_helper
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

        template <typename Executor, typename F, typename ... Ts>
        HPX_FORCEINLINE auto async_execute_fn::operator()(Executor const& exec,
                F && f, Ts &&... ts) const
        ->  decltype(async_execute_fn_helper<Executor>::call(
                exec, std::forward<F>(f), std::forward<Ts>(ts)...
            ))
        {
            return async_execute_fn_helper<Executor>::call(exec,
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

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

        template <typename Executor,
            typename std::enable_if<
                is_two_way_executor<Executor>::value
            >::type>
        struct sync_execute_fn_helper
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

        template <typename Executor, typename F, typename ... Ts>
        HPX_FORCEINLINE auto sync_execute_fn::operator()(Executor const& exec,
                F && f, Ts &&... ts) const
        ->  decltype(sync_execute_fn_helper<Executor>::call(
                exec, std::forward<F>(f), std::forward<Ts>(ts)...
            ))
        {
            return sync_execute_fn_helper<Executor>::call(exec,
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        ///////////////////////////////////////////////////////////////////////
        // default implementation of the then_execute() customization point
        template <typename TwoWayExecutor, typename F, typename Future,
            typename ... Ts, typename U =
                typename std::enable_if<
                    std::is_void<
                        typename hpx::traits::future_traits<Future>::type
                    >::value
                >::type>
        HPX_FORCEINLINE auto then_execute(TwoWayExecutor const& exec, F && f,
                Future& predecessor, Ts &&... ts)
        ->  decltype(exec.then_execute(
                std::forward<F>(f), predecessor, std::forward<Ts>(ts)...
            ))
        {
            return exec.then_execute(std::forward<F>(f), predecessor,
                std::forward<Ts>(ts)...);
        }

        template <typename TwoWayExecutor, typename F, typename Future,
            typename ... Ts, typename U =
                typename std::enable_if<
                   !std::is_void<
                        typename hpx::traits::future_traits<Future>::type
                    >::value
                >::type>
        HPX_FORCEINLINE auto then_execute(TwoWayExecutor const& exec, F && f,
                Future& predecessor, Ts &&... ts)
        ->  decltype(exec.then_execute(
                std::forward<F>(f), predecessor, std::forward<Ts>(ts)...
            ))
        {
            return exec.then_execute(std::forward<F>(f), predecessor,
                std::forward<Ts>(ts)...);
        }

        template <typename Executor,
            typename std::enable_if<
                is_two_way_executor<Executor>::value
            >::type>
        struct then_execute_fn_helper
        {
            template <typename TwoWayExecutor, typename F, typename Future,
                typename ... Ts>
            HPX_FORCEINLINE static auto call(TwoWayExecutor const& exec,
                    F && f, Future& predecessor, Ts &&... ts)
            ->  decltype(then_execute(
                    exec, std::forward<F>(f), predecessor, std::forward<Ts>(ts)...
                ))
            {
                return then_execute(exec, std::forward<F>(f), predecessor,
                    std::forward<Ts>(ts)...);
            }
        };

        template <typename Executor, typename F, typename Future,
            typename ... Ts>
        HPX_FORCEINLINE auto then_execute_fn::operator()(Executor const& exec,
                F && f, Future& predecessor, Ts &&... ts) const
        ->  decltype(then_execute_fn_helper<Executor>::call(
                exec, std::forward<F>(f), predecessor, std::forward<Ts>(ts)...
            ))
        {
            return then_execute_fn_helper<Executor>::call(exec,
                std::forward<F>(f), predecessor, std::forward<Ts>(ts)...);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // post()
    namespace detail
    {
        // default implementation of the post() customization point
        template <typename NonBlockingOneWayExecutor, typename F, typename ... Ts>
        auto post(NonBlockingOneWayExecutor const& exec, F && f, Ts &&... ts)
        ->  decltype(
                exec.post(std::forward<F>(f), std::forward<Ts>(ts)...)
            )
        {
            return exec.post(std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // dispatch to V1 executors
        template <typename NonBlockingOneWayExecutor, typename F, typename ... Ts>
        auto post(NonBlockingOneWayExecutor const& exec, F && f, Ts &&... ts)
        ->  decltype(
                exec.apply_execute(std::forward<F>(f), std::forward<Ts>(ts)...)
            )
        {
            return exec.apply_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename Executor,
            typename std::enable_if<
                is_non_blocking_one_way_executor<Executor>::value
            >::type>
        struct post_fn_helper
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
            auto operator()(Executor const& exec, F && f, Ts &&... ts) const
            ->  decltype(post_fn_helper<Executor>::call(
                    exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ))
            {
                return post_fn_helper<Executor>::call(exec,
                    std::forward<F>(f), std::forward<Ts>(ts)...);
            }
        };
    }
//
//     namespace
//     {
//         // execution::execute is a global function object
//         constexpr auto const& post =
//             detail::static_const<detail::post_fn>::value;
//     }
//
//     ///////////////////////////////////////////////////////////////////////////
//     // defer()
//     namespace detail
//     {
//         // default implementation of the defer() customization point
//         template <typename NonBlockingOneWayExecutor, typename F, typename ... Ts>
//         auto defer(NonBlockingOneWayExecutor const& exec, F && f, Ts &&... ts)
//         ->  decltype(
//                 exec.defer(std::forward<F>(f), std::forward<Ts>(ts)...)
//             )
//         {
//             return exec.defer(std::forward<F>(f), std::forward<Ts>(ts)...);
//         }
//
//         struct defer_fn
//         {
//             template <typename Executor, typename F, typename ... Ts>
//             auto operator()(Executor const& exec, F && f, Ts &&... ts) const
//             ->  decltype(
//                     defer(exec, std::forward<F>(f), std::forward<Ts>(ts)...)
//                 )
//             {
//                 return defer(exec, std::forward<F>(f), std::forward<Ts>(ts)...);
//             }
//         };
//     }
//
//     namespace
//     {
//         // execution::execute is a global function object
//         constexpr auto const& defer =
//             detail::static_const<detail::defer_fn>::value;
//     }

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

