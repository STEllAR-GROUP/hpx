//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nounnamed

/// \file parallel/executors/execution.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_EXECUTION_DEC_23_0712PM)
#define HPX_PARALLEL_EXECUTORS_EXECUTION_DEC_23_0712PM

#include <hpx/config.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/traits/detail/wrap_int.hpp>
#include <hpx/traits/future_then_result.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/detected.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/unwrapped.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/execution_fwd.hpp>

#include <iterator>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/range/const_iterator.hpp>
#include <boost/range/functions.hpp>
#include <boost/range/irange.hpp>
#include <boost/throw_exception.hpp>

#if defined(HPX_HAVE_CXX1Y_EXPERIMENTAL_OPTIONAL)
#include <experimental/optional>
#else
#include <boost/optional.hpp>
#endif

///////////////////////////////////////////////////////////////////////////////
// forward declare wait_all(std::vector<Future>&&) to avoid including
// wait_all.hpp which creates circular #include dependencies
namespace hpx { namespace lcos
{
    template <typename Future>
    void wait_all(std::vector<Future>&& values);
}}

///////////////////////////////////////////////////////////////////////////////
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
        template <typename Executor, typename ... Ts>
        struct async_execute_fn_not_callable;

        template <typename Executor, typename Enable = void>
        struct async_execute_fn_helper
        {
            template <typename OneWayExecutor, typename ... Ts>
            static auto call(OneWayExecutor&, Ts &&...)
            ->  async_execute_fn_not_callable<OneWayExecutor, Ts...>
            {
                return async_execute_fn_not_callable<OneWayExecutor, Ts...>{};
            }
        };

        template <typename Executor, typename ... Ts>
        struct sync_execute_fn_not_callable;

        template <typename Executor, typename Enable = void>
        struct sync_execute_fn_helper
        {
            template <typename OneWayExecutor, typename ... Ts>
            static auto call(OneWayExecutor&, Ts &&...)
            ->  sync_execute_fn_not_callable<OneWayExecutor, Ts...>
            {
                return sync_execute_fn_not_callable<OneWayExecutor, Ts...>{};
            }
        };

        template <typename Executor, typename ... Ts>
        struct then_execute_fn_not_callable;

        template <typename Executor, typename Enable = void>
        struct then_execute_fn_helper
        {
            template <typename OneWayExecutor, typename ... Ts>
            static auto call(OneWayExecutor&, Ts &&...)
            ->  then_execute_fn_not_callable<OneWayExecutor, Ts...>
            {
                return then_execute_fn_not_callable<OneWayExecutor, Ts...>{};
            }
        };

        template <typename Executor, typename ... Ts>
        struct post_fn_not_callable;

        template <typename Executor, typename Enable = void>
        struct post_fn_helper
        {
            template <typename BulkExecutor, typename ... Ts>
            static auto call(BulkExecutor&, Ts &&...)
            ->  post_fn_not_callable<BulkExecutor, Ts...>
            {
                return post_fn_not_callable<BulkExecutor, Ts...>{};
            }
        };

        template <typename Executor, typename ... Ts>
        struct async_bulk_execute_fn_not_callable;

        template <typename Executor, typename Enable = void>
        struct async_bulk_execute_fn_helper
        {
            template <typename BulkExecutor, typename ... Ts>
            static auto call(BulkExecutor&, Ts &&...)
            ->  async_bulk_execute_fn_not_callable<BulkExecutor, Ts...>
            {
                return async_bulk_execute_fn_not_callable<BulkExecutor, Ts...>{};
            }
        };

        template <typename Executor, typename ... Ts>
        struct sync_bulk_execute_fn_not_callable;

        template <typename Executor, typename Enable = void>
        struct sync_bulk_execute_fn_helper
        {
            template <typename BulkExecutor, typename ... Ts>
            static auto call(BulkExecutor&, Ts &&...)
            ->  sync_bulk_execute_fn_not_callable<BulkExecutor, Ts...>
            {
                return sync_bulk_execute_fn_not_callable<BulkExecutor, Ts...>{};
            }
        };

        template <typename Executor, typename Enable = void>
        struct then_bulk_execute_fn_helper;
    }

    // customization point for OneWayExecutor interface
    // execute()
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename F, typename ... Ts>
        struct sync_execute_not_callable;

        ///////////////////////////////////////////////////////////////////////
        // default implementation of the sync_execute() customization point
        template <typename Executor, typename F, typename ... Ts>
        HPX_FORCEINLINE auto sync_execute(hpx::traits::detail::wrap_int,
                Executor& exec, F && f, Ts &&... ts)
        ->  sync_execute_not_callable<Executor, F, Ts...>
        {
            return sync_execute_not_callable<Executor, F, Ts...>{};
        }

        template <typename OneWayExecutor, typename F, typename ... Ts>
        HPX_FORCEINLINE auto sync_execute(int, OneWayExecutor const& exec,
                F && f, Ts &&... ts)
        ->  decltype(
                exec.sync_execute(std::forward<F>(f), std::forward<Ts>(ts)...)
            )
        {
            return exec.sync_execute(std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

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
            HPX_FORCEINLINE static auto call_impl(std::false_type,
                    OneWayExecutor const& exec, F && f, Ts &&... ts)
            ->  hpx::future<decltype(sync_execute(
                    0, exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ))>
            {
                return hpx::lcos::make_ready_future(sync_execute(
                        0, exec, std::forward<F>(f), std::forward<Ts>(ts)...
                    ));
            }

            template <typename OneWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static hpx::future<void>
            call_impl(std::true_type,
                OneWayExecutor const& exec, F && f, Ts &&... ts)
            {
                sync_execute(0, exec, std::forward<F>(f),
                    std::forward<Ts>(ts)...);
                return hpx::lcos::make_ready_future();
            }

            template <typename OneWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static auto call(OneWayExecutor const& exec,
                    F && f, Ts &&... ts)
            ->  hpx::future<decltype(sync_execute(
                    0, exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ))>
            {
                typedef std::is_void<decltype(
                        sync_execute(0, exec, std::forward<F>(f),
                            std::forward<Ts>(ts)...)
                    )> is_void;

                return call_impl(is_void(), exec, std::forward<F>(f),
                    std::forward<Ts>(ts)...);
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
            ->  decltype(sync_execute(
                    0, exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ))
            {
                return sync_execute(0, exec, std::forward<F>(f),
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
            hpx::lcos::future<typename hpx::util::detail::deferred_result_of<
                F(Future, Ts...)
            >::type>
            call(OneWayExecutor const& exec, F && f, Future& predecessor,
                Ts &&... ts)
            {
                typedef typename hpx::util::detail::deferred_result_of<
                        F(Future, Ts...)
                    >::type result_type;

                auto func = hpx::util::bind(
                    hpx::util::one_shot(std::forward<F>(f)),
                    hpx::util::placeholders::_1, std::forward<Ts>(ts)...);

                typename hpx::traits::detail::shared_state_ptr<result_type>::type
                    p = lcos::detail::make_continuation_exec<result_type>(
                            predecessor, exec, std::move(func));

                return hpx::traits::future_access<hpx::lcos::future<result_type> >::
                    create(std::move(p));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // emulate post() on OneWayExecutors
        template <typename Executor>
        struct post_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value &&
               !hpx::traits::is_two_way_executor<Executor>::value &&
               !hpx::traits::is_non_blocking_one_way_executor<Executor>::value
            >::type>
        {
            // dispatch to V1 executors
            template <typename OneWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static auto
            call_impl(int, OneWayExecutor const& exec, F && f, Ts &&... ts)
            ->  decltype(exec.apply_execute(
                    std::forward<F>(f), std::forward<Ts>(ts)...
                ))
            {
                // use apply_execute, if exposed
                return exec.apply_execute(std::forward<F>(f),
                    std::forward<Ts>(ts)...);
            }

            template <typename OneWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static void
            call_impl(hpx::traits::detail::wrap_int, OneWayExecutor const& exec,
                F && f, Ts &&... ts)
            {
                // execute synchronously
                sync_execute(0, exec, std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            template <typename OneWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static auto call(OneWayExecutor const& exec,
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
    // customization points for TwoWayExecutor interface
    // async_execute(), sync_execute(), then_execute()
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename F, typename ... Ts>
        struct async_execute_not_callable;

        ///////////////////////////////////////////////////////////////////////
        // default implementation of the async_execute() customization point
        template <typename Executor, typename F, typename ... Ts>
        HPX_FORCEINLINE auto async_execute(hpx::traits::detail::wrap_int,
                Executor& exec, F && f, Ts &&... ts)
        ->  async_execute_not_callable<Executor, F, Ts...>
        {
            return async_execute_not_callable<Executor, F, Ts...>{};
        }

        template <typename TwoWayExecutor, typename F, typename ... Ts>
        HPX_FORCEINLINE auto async_execute(int,
                TwoWayExecutor const& exec, F && f, Ts &&... ts)
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
                    0, exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ))
            {
                return async_execute(0, exec, std::forward<F>(f),
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
        template <typename Executor>
        struct sync_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_two_way_executor<Executor>::value
            >::type>
        {
            // fall-back: emulate sync_execute using async_execute
            template <typename TwoWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static auto call_impl(std::false_type,
                    TwoWayExecutor const& exec, F && f, Ts &&... ts)
            ->  decltype(hpx::util::invoke(
                    std::forward<F>(f), std::forward<Ts>(ts)...
                ))
            {
                try {
                    typedef typename hpx::util::detail::deferred_result_of<
                            F(Ts...)
                        >::type result_type;

                    // older versions of gcc are not able to capture parameter
                    // packs (gcc < 4.9)
                    auto && args =
                        hpx::util::forward_as_tuple(std::forward<Ts>(ts)...);

#if defined(HPX_HAVE_CXX1Y_EXPERIMENTAL_OPTIONAL)
                    std::experimental::optional<result_type> out;
                    auto && wrapper =
                        [&]() mutable
                        {
                            out.emplace(hpx::util::invoke_fused(
                                std::forward<F>(f), std::move(args)));
                        };
#else
                    boost::optional<result_type> out;
                    auto && wrapper =
                        [&]() mutable
                        {
#if BOOST_VERSION < 105600
                            out = boost::in_place(hpx::util::invoke_fused(
                                std::forward<F>(f), std::move(args)));
#else
                            out.emplace(hpx::util::invoke_fused(
                                std::forward<F>(f), std::move(args)));
#endif
                        };
#endif

                    // use async execution, wait for result, propagate exceptions
                    async_execute(0, exec, std::ref(wrapper)).get();
                    return std::move(*out);
                }
                catch (std::bad_alloc const& ba) {
                    boost::throw_exception(ba);
                }
                catch (...) {
                    boost::throw_exception(
                        hpx::exception_list(boost::current_exception())
                    );
                }
            }

            template <typename TwoWayExecutor, typename F, typename ... Ts>
            static void call_impl(std::true_type,
                TwoWayExecutor const& exec, F && f, Ts &&... ts)
            {
                async_execute(
                    0, exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ).get();
            }

            template <typename TwoWayExecutor, typename F, typename ... Ts>
            static auto call_impl(hpx::traits::detail::wrap_int,
                    TwoWayExecutor const& exec, F && f, Ts &&... ts)
            ->  decltype(hpx::util::invoke(
                    std::forward<F>(f), std::forward<Ts>(ts)...
                ))
            {
                typedef typename std::is_void<
                        typename hpx::util::detail::deferred_result_of<
                            F(Ts...)
                        >::type
                    >::type is_void;

                return call_impl(is_void(), exec, std::forward<F>(f),
                    std::forward<Ts>(ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static auto call_impl(int,
                    TwoWayExecutor const& exec, F && f, Ts &&... ts)
            ->  decltype(exec.sync_execute(
                    std::forward<F>(f), std::forward<Ts>(ts)...
                ))
            {
                return exec.sync_execute(std::forward<F>(f),
                    std::forward<Ts>(ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename ... Ts>
            HPX_FORCEINLINE static auto call(TwoWayExecutor const& exec,
                    F && f, Ts &&... ts)
            ->  decltype(call_impl(
                    0, exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ))
            {
                return call_impl(0, exec, std::forward<F>(f),
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
        // then_execute()
        template <typename Executor>
        struct then_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_two_way_executor<Executor>::value
            >::type>
        {
            template <typename TwoWayExecutor, typename F, typename Future,
                typename ... Ts>
            HPX_FORCEINLINE static auto
            call_impl(int, TwoWayExecutor const& exec,
                    F && f, Future& predecessor, Ts &&... ts)
            ->  decltype(exec.then_execute(
                    std::forward<F>(f), predecessor, std::forward<Ts>(ts)...
                ))
            {
                return exec.then_execute(std::forward<F>(f),
                    predecessor, std::forward<Ts>(ts)...);
            }

            template <typename TwoWayExecutor, typename F, typename Future,
                typename ... Ts>
            HPX_FORCEINLINE static auto
            call_impl(hpx::traits::detail::wrap_int,
                    TwoWayExecutor const& exec, F && f, Future& predecessor,
                    Ts &&... ts)
            {
                typedef typename hpx::util::detail::deferred_result_of<
                        F(Future, Ts...)
                    >::type result_type;

                auto func = hpx::util::bind(
                    hpx::util::one_shot(std::forward<F>(f)),
                    hpx::util::placeholders::_1, std::forward<Ts>(ts)...);

                typename hpx::traits::detail::shared_state_ptr<result_type>::type
                    p = lcos::detail::make_continuation_exec<result_type>(
                            predecessor, exec, std::move(func));

                return hpx::traits::future_access<hpx::lcos::future<result_type> >::
                    create(std::move(p));
            }

            template <typename TwoWayExecutor, typename F, typename Future,
                typename ... Ts>
            HPX_FORCEINLINE static auto call(TwoWayExecutor const& exec,
                    F && f, Future& predecessor, Ts &&... ts)
            ->  decltype(call_impl(
                    0, exec, std::forward<F>(f), predecessor,
                    std::forward<Ts>(ts)...
                ))
            {
                return call_impl(0, exec, std::forward<F>(f),
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
                async_execute(0, exec, std::forward<F>(f),
                    std::forward<Ts>(ts)...);
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
        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename F, typename ... Ts>
        struct post_not_callable;

        ///////////////////////////////////////////////////////////////////////
        // default implementation of the async_execute() customization point
        template <typename Executor, typename F, typename ... Ts>
        HPX_FORCEINLINE auto post(hpx::traits::detail::wrap_int,
                Executor& exec, F && f, Ts &&... ts)
        ->  post_not_callable<Executor, F, Ts...>
        {
            return post_not_callable<Executor, F, Ts...>{};
        }

        // default implementation of the post() customization point
        template <typename NonBlockingOneWayExecutor, typename F,
            typename ... Ts>
        HPX_FORCEINLINE auto post(int,
                NonBlockingOneWayExecutor const& exec, F && f, Ts &&... ts)
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
                    0, exec, std::forward<F>(f), std::forward<Ts>(ts)...
                ))
            {
                return post(0, exec, std::forward<F>(f),
                    std::forward<Ts>(ts)...);
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
            static T const value;
        };

        template <typename T>
        T const static_const<T>::value = T{};
    }
    /// \endcond

    namespace
    {
        ///////////////////////////////////////////////////////////////////////
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
        ///       exec.sync_execute(f, ts...) if it exists.
        ///
        constexpr detail::sync_execute_fn const& sync_execute =
            detail::static_const<detail::sync_execute_fn>::value;

        ///////////////////////////////////////////////////////////////////////
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
        ///       make_ready_future(exec.sync_execute(f, ts...) if it exists)
        ///       and for two way executors (calls exec.async_execute(f, ts...)
        ///       if it exists).
        ///
        /// \returns f(ts...)'s result through a future
        ///
        constexpr detail::async_execute_fn const& async_execute =
            detail::static_const<detail::async_execute_fn>::value;

        /// Customization point for execution agent creation depending on a
        /// given future.
        ///
        /// This creates a single function invocation f() using the associated
        /// executor after the given future object has become ready.
        ///
        /// \param exec [in] The executor object to use for scheduling of the
        ///             function \a f.
        /// \param f    [in] The function which will be scheduled using the
        ///             given executor.
        /// \param predecessor [in] The future object the execution of the
        ///             given function depends on.
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

        ///////////////////////////////////////////////////////////////////////
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

    /// \cond NOINTERNAL
    ///////////////////////////////////////////////////////////////////////////
    // customization points for BulkTwoWayExecutor interface
    // async_bulk_execute(), sync_bulk_execute(), then_bulk_execute()
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // async_bulk_execute()
        template <typename F, typename Shape, typename ... Ts>
        struct bulk_function_result
        {
            typedef typename
                    boost::range_const_iterator<Shape>::type
                iterator_type;
            typedef typename
                    std::iterator_traits<iterator_type>::value_type
                value_type;
            typedef typename
                    hpx::util::detail::deferred_result_of<
                        F(value_type, Ts...)
                    >::type
                type;
        };

        template <typename Executor>
        struct async_bulk_execute_fn_helper<Executor,
            typename std::enable_if<
               (hpx::traits::is_one_way_executor<Executor>::value ||
                    hpx::traits::is_two_way_executor<Executor>::value) &&
               !hpx::traits::is_bulk_two_way_executor<Executor>::value
            >::type>
        {
            template <typename BulkExecutor, typename F, typename Shape,
                typename ... Ts>
            HPX_FORCEINLINE static auto
            call_impl(hpx::traits::detail::wrap_int,
                    BulkExecutor const& exec, F && f, Shape const& shape,
                    Ts &&... ts)
            ->  std::vector<typename hpx::traits::executor_future<
                        Executor,
                        typename bulk_function_result<F, Shape, Ts...>::type,
                        Ts...
                    >::type>
            {
                std::vector<typename hpx::traits::executor_future<
                        Executor,
                        typename bulk_function_result<F, Shape, Ts...>::type,
                        Ts...
                    >::type> results;

// Before Boost V1.56 boost::size() does not respect the iterator category of
// its argument.
#if BOOST_VERSION < 105600
                results.reserve(
                    std::distance(boost::begin(shape), boost::end(shape)));
#else
                results.reserve(boost::size(shape));
#endif

                for (auto const& elem: shape)
                {
                    results.push_back(
                        execution::async_execute(exec, f, elem, ts...)
                    );
                }

                return results;
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename ... Ts>
            HPX_FORCEINLINE static auto
            call_impl(int, BulkExecutor const& exec, F && f,
                    Shape const& shape, Ts &&... ts)
            ->  decltype(exec.async_bulk_execute(
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...
                ))
            {
                return exec.async_bulk_execute(std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename ... Ts>
            HPX_FORCEINLINE static auto
            call(BulkExecutor const& exec, F && f, Shape const& shape,
                    Ts &&... ts)
            ->  decltype(call_impl(
                    0, exec, std::forward<F>(f), shape, std::forward<Ts>(ts)...
                ))
            {
                return call_impl(0, exec, std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...);
            }
        };

        template <typename Executor>
        struct async_bulk_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_bulk_two_way_executor<Executor>::value
            >::type>
        {
            template <typename BulkExecutor, typename F, typename Shape,
                typename ... Ts>
            HPX_FORCEINLINE static auto
            call(BulkExecutor const& exec, F && f, Shape const& shape,
                    Ts &&... ts)
//             ->  decltype(exec.async_bulk_execute(
//                     std::forward<F>(f), shape, std::forward<Ts>(ts)...
//                 ))
            {
                return exec.async_bulk_execute(std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...);
            }
        };

        // async_bulk_execute dispatch point
        struct async_bulk_execute_fn
        {
            template <typename Executor, typename F, typename Shape,
                typename ... Ts>
            HPX_FORCEINLINE auto operator()(Executor const& exec, F && f,
                    Shape const& shape, Ts &&... ts) const
//             ->  decltype(async_bulk_execute_fn_helper<Executor>::call(
//                     exec, std::forward<F>(f), shape, std::forward<Ts>(ts)...
//                 ))
            {
                return async_bulk_execute_fn_helper<Executor>::call(exec,
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // sync_bulk_execute()
        template <typename F, typename Shape, bool IsVoid, typename ... Ts>
        struct bulk_execute_result_impl;

        template <typename F, typename Shape, typename ... Ts>
        struct bulk_execute_result_impl<F, Shape, false, Ts...>
        {
            typedef std::vector<
                    typename bulk_function_result<F, Shape, Ts...>::type
                > type;
        };

        template <typename F, typename Shape, typename ... Ts>
        struct bulk_execute_result_impl<F, Shape, true, Ts...>
        {
            typedef void type;
        };

        template <typename F, typename Shape, typename ... Ts>
        struct bulk_execute_result
          : bulk_execute_result_impl<F, Shape,
                std::is_void<
                    typename bulk_function_result<F, Shape, Ts...>::type
                >::value,
                Ts...>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor>
        struct sync_bulk_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value &&
               !hpx::traits::is_two_way_executor<Executor>::value &&
               !hpx::traits::is_bulk_one_way_executor<Executor>::value
            >::type>
        {
            // returns void if F returns void
            template <typename BulkExecutor, typename F, typename Shape,
                typename ... Ts>
            static auto call_impl(std::false_type,
                    BulkExecutor const& exec, F && f, Shape const& shape,
                    Ts &&... ts)
            ->  typename bulk_execute_result<F, Shape, Ts...>::type
            {
                try {
                    typename bulk_execute_result<F, Shape, Ts...>::type results;

// Before Boost V1.56 boost::size() does not respect the iterator category of
// its argument.
#if BOOST_VERSION < 105600
                    results.reserve(
                        std::distance(boost::begin(shape), boost::end(shape)));
#else
                    results.reserve(boost::size(shape));
#endif

                    for (auto const& elem : shape)
                    {
                        results.push_back(
                            execution::sync_execute(exec, f, elem, ts...)
                        );
                    }
                    return results;
                }
                catch (std::bad_alloc const& ba) {
                    boost::throw_exception(ba);
                }
                catch (...) {
                    boost::throw_exception(
                        exception_list(boost::current_exception())
                    );
                }
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename ... Ts>
            static void call_impl(std::true_type,
                BulkExecutor const& exec, F && f, Shape const& shape,
                Ts &&... ts)
            {
                try {
                    for (auto const& elem : shape)
                    {
                        execution::sync_execute(exec, f, elem, ts...);
                    }
                }
                catch (std::bad_alloc const& ba) {
                    boost::throw_exception(ba);
                }
                catch (...) {
                    boost::throw_exception(
                        exception_list(boost::current_exception())
                    );
                }
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename ... Ts>
            static auto call_impl(hpx::traits::detail::wrap_int,
                    BulkExecutor const& exec, F && f, Shape const& shape,
                    Ts &&... ts)
            ->  typename bulk_execute_result<F, Shape, Ts...>::type
            {
                typedef typename std::is_void<
                        typename bulk_function_result<F, Shape, Ts...>::type
                    >::type is_void;

                return call_impl(is_void(), exec, std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename ... Ts>
            static auto call_impl(int,
                    BulkExecutor const& exec, F && f, Shape const& shape,
                    Ts &&... ts)
            ->  decltype(exec.sync_bulk_execute(
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...
                ))
            {
                return exec.sync_bulk_execute(std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename ... Ts>
            HPX_FORCEINLINE static auto
            call(BulkExecutor const& exec, F && f, Shape const& shape,
                    Ts &&... ts)
            ->  decltype(call_impl(
                    0, exec, std::forward<F>(f), shape, std::forward<Ts>(ts)...
                ))
            {
                return call_impl(0, exec, std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...);
            }
        };

        template <typename Executor>
        struct sync_bulk_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_two_way_executor<Executor>::value &&
               !hpx::traits::is_bulk_one_way_executor<Executor>::value
            >::type>
        {
            template <typename BulkExecutor, typename F, typename Shape,
                typename ... Ts>
            static auto call_impl(std::false_type,
                    BulkExecutor const& exec, F && f, Shape const& shape,
                    Ts &&... ts)
            ->  typename bulk_execute_result<F, Shape, Ts...>::type
            {
                typedef typename hpx::traits::executor_future<
                        Executor,
                        typename bulk_execute_result<F, Shape, Ts...>::type
                    >::type result_type;

                try {
// Before Boost V1.56 boost::size() does not respect the iterator category of
// its argument.
#if BOOST_VERSION < 105600
                    result_type results;
                    results.reserve(
                        std::distance(boost::begin(shape), boost::end(shape)));
#else
                    result_type results;
                    results.reserve(boost::size(shape));
#endif
                    for (auto const& elem : shape)
                    {
                        results.push_back(
                            execution::async_execute(exec, f, elem, ts...)
                        );
                    }
                    return hpx::util::unwrapped(results);
                }
                catch (std::bad_alloc const& ba) {
                    boost::throw_exception(ba);
                }
                catch (...) {
                    boost::throw_exception(
                        exception_list(boost::current_exception())
                    );
                }
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename ... Ts>
            static void call_impl(std::true_type,
                    BulkExecutor const& exec, F && f, Shape const& shape,
                    Ts &&... ts)
            {
                typedef std::vector<
                        typename hpx::traits::executor_future<
                            Executor,
                            typename bulk_function_result<F, Shape, Ts...>::type
                        >::type
                    > result_type;

                try {
// Before Boost V1.56 boost::size() does not respect the iterator category of
// its argument.
#if BOOST_VERSION < 105600
                    result_type results;
                    results.reserve(
                        std::distance(boost::begin(shape), boost::end(shape)));
#else
                    result_type results;
                    results.reserve(boost::size(shape));
#endif

                    for (auto const& elem : shape)
                    {
                        results.push_back(
                            execution::async_execute(exec, f, elem, ts...)
                        );
                    }
                    hpx::lcos::wait_all(std::move(results));
                }
                catch (std::bad_alloc const& ba) {
                    boost::throw_exception(ba);
                }
                catch (...) {
                    boost::throw_exception(
                        exception_list(boost::current_exception())
                    );
                }
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename ... Ts>
            static auto call_impl(hpx::traits::detail::wrap_int,
                    BulkExecutor const& exec, F && f, Shape const& shape,
                    Ts &&... ts)
            ->  typename bulk_execute_result<F, Shape, Ts...>::type
            {
                typedef typename std::is_void<
                        typename bulk_function_result<F, Shape, Ts...>::type
                    >::type is_void;

                return call_impl(is_void(), exec, std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename ... Ts>
            static auto call_impl(int,
                    BulkExecutor const& exec, F && f, Shape const& shape,
                    Ts &&... ts)
            ->  decltype(exec.sync_bulk_execute(
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...
                ))
            {
                return exec.sync_bulk_execute(std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename ... Ts>
            HPX_FORCEINLINE static auto
            call(BulkExecutor const& exec, F && f, Shape const& shape,
                    Ts &&... ts)
            ->  decltype(call_impl(
                    0, exec, std::forward<F>(f), shape, std::forward<Ts>(ts)...
                ))
            {
                return call_impl(0, exec, std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...);
            }
        };

        template <typename Executor>
        struct sync_bulk_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_bulk_one_way_executor<Executor>::value
            >::type>
        {
            template <typename BulkExecutor, typename F, typename Shape,
                typename ... Ts>
            static auto
            call(BulkExecutor const& exec, F && f, Shape const& shape,
                    Ts &&... ts)
            ->  decltype(exec.sync_bulk_execute(
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...
                ))
            {
                return exec.sync_bulk_execute(std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...);
            }
        };

        struct sync_bulk_execute_fn
        {
            template <typename Executor, typename F, typename Shape,
                typename ... Ts>
            HPX_FORCEINLINE auto operator()(Executor const& exec, F && f,
                    Shape const& shape, Ts &&... ts) const
            ->  decltype(sync_bulk_execute_fn_helper<Executor>::call(
                    exec, std::forward<F>(f), shape, std::forward<Ts>(ts)...
                ))
            {
                return sync_bulk_execute_fn_helper<Executor>::call(exec,
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...);
            }
        };
    }
    /// \endcond

    namespace
    {
        ///////////////////////////////////////////////////////////////////////
        // BulkTwoWayExecutor customization points:
        // execution::async_bulk_execute, execution::sync_bulk_execute,
        // execution::then_bulk_execute

        /// Bulk form of asynchronous execution agent creation.
        ///
        /// \note This is deliberately different from the bulk_async_execute
        ///       customization points specified in P0443.The async_bulk_execute
        ///       customization point defined here is more generic and is used
        ///       as the workhorse for implementing the specified APIs.
        ///
        /// This asynchronously creates a group of function invocations f(i)
        /// whose ordering is given by the execution_category associated with
        /// the executor.
        ///
        /// Here \a i takes on all values in the index space implied by shape.
        /// All exceptions thrown by invocations of f(i) are reported in a
        /// manner consistent with parallel algorithm execution through the
        /// returned future.
        ///
        /// \param exec  [in] The executor object to use for scheduling of the
        ///              function \a f.
        /// \param f     [in] The function which will be scheduled using the
        ///              given executor.
        /// \param shape [in] The shape objects which defines the iteration
        ///              boundaries for the arguments to be passed to \a f.
        /// \param ts    [in] Additional arguments to use to invoke \a f.
        ///
        /// \returns The return type of \a executor_type::async_bulk_execute if
        ///          defined by \a executor_type. Otherwise a vector
        ///          of futures holding the returned values of each invocation
        ///          of \a f.
        ///
        /// \note This calls exec.async_bulk_execute(f, shape, ts...) if it
        ///       exists; otherwise it executes async_execute(f, shape, ts...)
        ///       as often as needed.
        ///
        constexpr detail::async_bulk_execute_fn const& async_bulk_execute =
            detail::static_const<detail::async_bulk_execute_fn>::value;

        /// Bulk form of synchronous execution agent creation.
        ///
        /// \note This is deliberately different from the bulk_sync_execute
        ///       customization points specified in P0443.The sync_bulk_execute
        ///       customization point defined here is more generic and is used
        ///       as the workhorse for implementing the specified APIs.
        ///
        /// This synchronously creates a group of function invocations f(i)
        /// whose ordering is given by the execution_category associated with
        /// the executor. The function synchronizes the execution of all
        /// scheduled functions with the caller.
        ///
        /// Here \a i takes on all values in the index space implied by shape.
        /// All exceptions thrown by invocations of f(i) are reported in a
        /// manner consistent with parallel algorithm execution through the
        /// returned future.
        ///
        /// \param exec  [in] The executor object to use for scheduling of the
        ///              function \a f.
        /// \param f     [in] The function which will be scheduled using the
        ///              given executor.
        /// \param shape [in] The shape objects which defines the iteration
        ///              boundaries for the arguments to be passed to \a f.
        /// \param ts    [in] Additional arguments to use to invoke \a f.
        ///
        /// \returns The return type of \a executor_type::sync_bulk_execute
        ///          if defined by \a executor_type. Otherwise a vector holding
        ///          the returned values of each invocation of \a f except when
        ///          \a f returns void, which case void is returned.
        ///
        /// \note This calls exec.sync_bulk_execute(f, shape, ts...) if it
        ///       exists; otherwise it executes sync_execute(f, shape, ts...)
        ///       as often as needed.
        ///
        constexpr detail::sync_bulk_execute_fn const& sync_bulk_execute =
            detail::static_const<detail::sync_bulk_execute_fn>::value;
    }

    /// \cond NOINTERNAL
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // then_bulk_execute()
        template <typename F, typename Shape, typename Future, typename ... Ts>
        struct then_bulk_function_result
        {
            typedef typename
                    boost::range_const_iterator<Shape>::type
                iterator_type;
            typedef typename
                    std::iterator_traits<iterator_type>::value_type
                value_type;
            typedef typename
                    hpx::util::detail::deferred_result_of<
                        F(value_type, Future, Ts...)
                    >::type
                type;
        };

        template <typename F, typename Shape, typename Future, bool IsVoid,
            typename ... Ts>
        struct then_bulk_execute_result_impl;

        template <typename F, typename Shape, typename Future, typename ... Ts>
        struct then_bulk_execute_result_impl<F, Shape, Future, false, Ts...>
        {
            typedef std::vector<
                    typename then_bulk_function_result<
                        F, Shape, Future, Ts...
                    >::type
                > type;
        };

        template <typename F, typename Shape, typename Future, typename ... Ts>
        struct then_bulk_execute_result_impl<F, Shape, Future, true, Ts...>
        {
            typedef void type;
        };

        template <typename F, typename Shape, typename Future, typename ... Ts>
        struct then_bulk_execute_result
          : then_bulk_execute_result_impl<F, Shape, Future,
                std::is_void<
                    typename then_bulk_function_result<
                        F, Shape, Future, Ts...
                    >::type
                >::value,
                Ts...>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename F, typename Shape,
            typename Future, std::size_t ... Is, typename ... Ts>
        HPX_FORCEINLINE auto
        fused_sync_bulk_execute(Executor const& exec,
                F && f, Shape const& shape, Future& predecessor,
                hpx::util::detail::pack_c<std::size_t, Is...>,
                hpx::util::tuple<Ts...> const& args)
        ->  decltype(execution::sync_bulk_execute(exec,
                std::forward<F>(f), shape, predecessor,
                    std::move(hpx::util::get<Is>(args))...
            ))
        {
            return execution::sync_bulk_execute(exec, std::forward<F>(f),
                shape, predecessor, std::move(hpx::util::get<Is>(args))...);
        }

        template <typename Executor>
        struct then_bulk_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value &&
               !hpx::traits::is_bulk_two_way_executor<Executor>::value
            >::type>
        {
            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename ... Ts>
            static auto
            call_impl(std::false_type, BulkExecutor const& exec,
                    F && f, Shape const& shape, Future predecessor,
                    Ts &&... ts)
            ->  hpx::future<typename then_bulk_execute_result<
                        F, Shape, Future, Ts...
                    >::type>
            {
                typedef typename then_bulk_execute_result<
                        F, Shape, Future, Ts...
                    >::type result_type;

                // older versions of gcc are not able to capture parameter
                // packs (gcc < 4.9)
                auto args = hpx::util::make_tuple(std::forward<Ts>(ts)...);
                auto func =
                    [exec, f, shape, args](Future predecessor)
                    ->  result_type
                    {
                        return fused_sync_bulk_execute(
                                exec, f, shape, predecessor,
                                typename hpx::util::detail::make_index_pack<
                                    sizeof...(Ts)
                                >::type(), args);
                    };

                typedef typename hpx::traits::detail::shared_state_ptr<
                        result_type
                    >::type shared_state_type;

                shared_state_type p =
                    lcos::detail::make_continuation_exec<result_type>(
                        predecessor, exec, std::move(func));

                return hpx::traits::future_access<hpx::future<result_type> >::
                    create(std::move(p));
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename ... Ts>
            HPX_FORCEINLINE static hpx::future<void>
            call_impl(std::true_type, BulkExecutor const& exec,
                    F && f, Shape const& shape, Future predecessor,
                    Ts &&... ts)
            {
                // older versions of gcc are not able to capture parameter
                // packs (gcc < 4.9)
                auto args = hpx::util::make_tuple(std::forward<Ts>(ts)...);
                auto func =
                    [exec, f, shape, args](Future predecessor) -> void
                    {
                        fused_sync_bulk_execute(
                            exec, f, shape, predecessor,
                            typename hpx::util::detail::make_index_pack<
                                sizeof...(Ts)
                            >::type(), args);
                    };

                typename hpx::traits::detail::shared_state_ptr<void>::type p =
                    lcos::detail::make_continuation_exec<void>(
                        predecessor, exec, std::move(func));

                return hpx::traits::future_access<hpx::future<void> >::
                    create(std::move(p));
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename ... Ts>
            HPX_FORCEINLINE static auto
            call_impl(hpx::traits::detail::wrap_int, BulkExecutor const& exec,
                    F && f, Shape const& shape, Future predecessor,
                    Ts &&... ts)
            ->  hpx::future<typename then_bulk_execute_result<
                        F, Shape, Future, Ts...
                    >::type>
            {
                typedef typename std::is_void<
                        typename then_bulk_function_result<
                            F, Shape, Future, Ts...
                        >::type
                    >::type is_void;

                return then_bulk_execute_fn_helper::call_impl(is_void(),
                    exec, std::forward<F>(f), shape, predecessor,
                    std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename ... Ts>
            HPX_FORCEINLINE static auto
            call_impl(int, BulkExecutor const& exec, F && f, Shape const& shape,
                    Future& predecessor, Ts &&... ts)
            ->  decltype(exec.then_bulk_execute(
                    std::forward<F>(f), shape, predecessor,
                        std::forward<Ts>(ts)...
                ))
            {
                return exec.then_bulk_execute(std::forward<F>(f), shape,
                    predecessor, std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename ... Ts>
            HPX_FORCEINLINE static auto
            call(BulkExecutor const& exec, F && f, Shape const& shape,
                    Future& predecessor, Ts &&... ts)
            ->  decltype(call_impl(
                    0, exec, std::forward<F>(f), shape, predecessor,
                    std::forward<Ts>(ts)...
                ))
            {
                return call_impl(0, exec, std::forward<F>(f), shape,
                    hpx::lcos::make_shared_future(predecessor),
                    std::forward<Ts>(ts)...);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename F, typename Shape,
            typename Future, std::size_t ... Is, typename ... Ts>
        HPX_FORCEINLINE auto
        fused_async_bulk_execute(Executor const& exec,
                F && f, Shape const& shape, Future& predecessor,
                hpx::util::detail::pack_c<std::size_t, Is...>,
                hpx::util::tuple<Ts...> const& args)
        ->  decltype(execution::async_bulk_execute(
                exec, std::forward<F>(f), shape, predecessor,
                    std::move(hpx::util::get<Is>(args))...
            ))
        {
            return execution::async_bulk_execute(exec, std::forward<F>(f),
                shape, predecessor, std::move(hpx::util::get<Is>(args))...);
        }

        template <typename Executor>
        struct then_bulk_execute_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_bulk_two_way_executor<Executor>::value
            >::type>
        {
            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename ... Ts>
            HPX_FORCEINLINE static auto
            call_impl(hpx::traits::detail::wrap_int, BulkExecutor const& exec,
                    F && f, Shape const& shape, Future predecessor,
                    Ts &&... ts)
            ->  typename hpx::traits::executor_future<
                    Executor,
                    typename then_bulk_function_result<
                        F, Shape, Future, Ts...
                    >::type
                >::type
            {
                typedef typename then_bulk_function_result<
                        F, Shape, Future, Ts...
                    >::type func_result_type;

                typedef std::vector<typename hpx::traits::executor_future<
                        Executor, func_result_type, Ts...
                    >::type> result_type;

                typedef typename hpx::traits::executor_future<
                        Executor, result_type
                    >::type result_future_type;

                // older versions of gcc are not able to capture parameter
                // packs (gcc < 4.9)
                auto args = hpx::util::make_tuple(std::forward<Ts>(ts)...);
                auto func =
                    [exec, f, shape, args](Future predecessor)
                    ->  result_type
                    {
                        return fused_async_bulk_execute(
                            exec, f, shape, predecessor,
                            typename hpx::util::detail::make_index_pack<
                                sizeof...(Ts)
                            >::type(), args);
                    };

                typedef typename hpx::traits::detail::shared_state_ptr<
                        result_type
                    >::type shared_state_type;

                shared_state_type p =
                    lcos::detail::make_continuation_exec<result_type>(
                        predecessor, exec, std::move(func));

                return hpx::traits::future_access<result_future_type>::
                    create(std::move(p));
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename ... Ts>
            HPX_FORCEINLINE static auto
            call_impl(int, BulkExecutor const& exec, F && f,
                    Shape const& shape, Future& predecessor, Ts &&... ts)
            ->  decltype(exec.then_bulk_execute(
                    std::forward<F>(f), shape, predecessor,
                        std::forward<Ts>(ts)...
                ))
            {
                return exec.then_bulk_execute(std::forward<F>(f), shape,
                    predecessor, std::forward<Ts>(ts)...);
            }

            template <typename BulkExecutor, typename F, typename Shape,
                typename Future, typename ... Ts>
            HPX_FORCEINLINE static auto
            call(BulkExecutor const& exec, F && f, Shape const& shape,
                    Future& predecessor, Ts &&... ts)
            ->  decltype(call_impl(
                    0, exec, std::forward<F>(f), shape, predecessor,
                    std::forward<Ts>(ts)...
                ))
            {
                return call_impl(0, exec, std::forward<F>(f), shape,
                    hpx::lcos::make_shared_future(predecessor),
                    std::forward<Ts>(ts)...);
            }
        };

        struct then_bulk_execute_fn
        {
            template <typename Executor, typename F, typename Shape,
                typename Future, typename ... Ts>
            HPX_FORCEINLINE auto operator()(Executor const& exec, F && f,
                    Shape const& shape, Future& predecessor, Ts &&... ts) const
            ->  decltype(then_bulk_execute_fn_helper<Executor>::call(
                    exec, std::forward<F>(f), shape, predecessor,
                    std::forward<Ts>(ts)...
                ))
            {
                return then_bulk_execute_fn_helper<Executor>::call(
                    exec, std::forward<F>(f), shape, predecessor,
                    std::forward<Ts>(ts)...);
            }
        };
    }
    /// \endcond

    namespace
    {
        ///////////////////////////////////////////////////////////////////////
        // BulkTwoWayExecutor customization points:
        // execution::then_bulk_execute

        /// Bulk form of execution agent creation depending on a given future.
        ///
        /// \note This is deliberately different from the then_sync_execute
        ///       customization points specified in P0443.The then_bulk_execute
        ///       customization point defined here is more generic and is used
        ///       as the workhorse for implementing the specified APIs.
        ///
        /// This creates a group of function invocations f(i)
        /// whose ordering is given by the execution_category associated with
        /// the executor.
        ///
        /// Here \a i takes on all values in the index space implied by shape.
        /// All exceptions thrown by invocations of f(i) are reported in a
        /// manner consistent with parallel algorithm execution through the
        /// returned future.
        ///
        /// \param exec  [in] The executor object to use for scheduling of the
        ///              function \a f.
        /// \param f     [in] The function which will be scheduled using the
        ///              given executor.
        /// \param shape [in] The shape objects which defines the iteration
        ///              boundaries for the arguments to be passed to \a f.
        /// \param predecessor [in] The future object the execution of the
        ///             given function depends on.
        /// \param ts    [in] Additional arguments to use to invoke \a f.
        ///
        /// \returns The return type of \a executor_type::then_bulk_execute
        ///          if defined by \a executor_type. Otherwise a vector holding
        ///          the returned values of each invocation of \a f.
        ///
        /// \note This calls exec.then_bulk_execute(f, shape, pred, ts...) if it
        ///       exists; otherwise it executes
        ///       sync_execute(f, shape, pred.share(), ts...) (if this executor
        ///       is also an OneWayExecutor), or
        ///       async_execute(f, shape, pred.share(), ts...) (if this executor
        ///       is also a TwoWayExecutor) - as often as needed.
        ///
        constexpr detail::then_bulk_execute_fn const& then_bulk_execute =
            detail::static_const<detail::then_bulk_execute_fn>::value;
    }
}}}}

#endif

