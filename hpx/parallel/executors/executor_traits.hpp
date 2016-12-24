//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/executor_traits.hpp

#if !defined(HPX_PARALLEL_EXECUTOR_TRAITS_MAY_10_2015_1128AM)
#define HPX_PARALLEL_EXECUTOR_TRAITS_MAY_10_2015_1128AM

#include <hpx/config.hpp>
#include <hpx/async.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/traits/detail/wrap_int.hpp>
#include <hpx/traits/is_executor_v1.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/unwrapped.hpp>

#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/rebind_executor.hpp>

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

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    HPX_STATIC_CONSTEXPR task_execution_policy_tag task{};

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename Enable = void>
        struct execution_category
        {
            typedef parallel_execution_tag type;
        };

        template <typename Executor>
        struct execution_category<Executor,
            typename hpx::util::always_void<
                typename Executor::execution_category
            >::type>
        {
            typedef typename Executor::execution_category type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename T, typename Enable = void>
        struct future_type
        {
            typedef hpx::future<T> type;
        };

        template <typename Executor, typename T>
        struct future_type<Executor, T,
            typename hpx::util::always_void<
                typename Executor::template future_type<T>::type
            >::type>
        {
            typedef typename Executor::template future_type<T>::type type;
        };

        ///////////////////////////////////////////////////////////////////////
        struct apply_helper
        {
            template <typename Executor, typename F, typename ... Ts>
            static void call(hpx::traits::detail::wrap_int, Executor&& exec,
                F && f, Ts &&... ts)
            {
                exec.async_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
            }

            template <typename Executor, typename F, typename ... Ts>
            static auto call(int, Executor&& exec, F && f, Ts &&... ts)
            ->  decltype(
                    exec.apply_execute(std::forward<F>(f), std::forward<Ts>(ts)...)
                )
            {
                exec.apply_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
            }
        };

        template <typename Executor, typename F, typename ... Ts>
        void call_apply_execute(Executor&& exec, F && f, Ts && ... ts)
        {
            apply_helper::call(0, std::forward<Executor>(exec),
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        ///////////////////////////////////////////////////////////////////////
        struct execute_helper
        {
            template <typename Executor, typename F, typename ... Ts>
            static auto call_impl(Executor&& exec, F && f, std::false_type,
                    Ts &&... ts)
            ->  decltype(
                    hpx::util::invoke(std::forward<F>(f), std::forward<Ts>(ts)...)
                )
            {
                try {
                    typedef typename hpx::util::detail::deferred_result_of<
                            F(Ts&&...)
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
                    exec.async_execute(std::ref(wrapper)).get();
                    return std::move(*out);
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

            template <typename Executor, typename F, typename ... Ts>
            static void call_impl(Executor&& exec, F && f, std::true_type,
                Ts &&... ts)
            {
                exec.async_execute(
                    std::forward<F>(f), std::forward<Ts>(ts)...
                ).get();
            }

            template <typename Executor, typename F, typename ... Ts>
            static auto call(hpx::traits::detail::wrap_int, Executor&& exec,
                    F && f, Ts &&... ts)
            ->  decltype(
                    hpx::util::invoke(std::forward<F>(f), std::forward<Ts>(ts)...)
                )
            {
                typedef std::is_void<
                        typename hpx::util::detail::deferred_result_of<
                            F(Ts&&...)
                        >::type
                    > is_void;
                return call_impl(std::forward<Executor>(exec),
                    std::forward<F>(f), is_void(), std::forward<Ts>(ts)...);
            }

            template <typename Executor, typename F, typename ... Ts>
            static auto call(int, Executor&& exec, F && f, Ts &&... ts)
            ->  decltype(
                    exec.execute(std::forward<F>(f), std::forward<Ts>(ts)...)
                )
            {
                return exec.execute(std::forward<F>(f),
                    std::forward<Ts>(ts)...);
            }
        };

        template <typename Executor, typename F, typename ... Ts>
        auto call_execute(Executor&& exec, F && f, Ts &&... ts)
        ->  decltype(
                execute_helper::call(0, std::forward<Executor>(exec),
                    std::forward<F>(f), std::forward<Ts>(ts)...)
            )
        {
            return execute_helper::call(0, std::forward<Executor>(exec),
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Shape, typename ... Ts>
        struct bulk_async_execute_result
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

        ///////////////////////////////////////////////////////////////////////
        struct bulk_async_execute_helper
        {
            template <typename Executor, typename F, typename S, typename ... Ts>
            static auto
            call(hpx::traits::detail::wrap_int, Executor&& exec, F && f,
                    S const& shape, Ts &&... ts)
            ->  std::vector<typename future_type<
                        typename hpx::util::decay<Executor>::type,
                        typename bulk_async_execute_result<F, S, Ts...>::type
                    >::type>
            {
                std::vector<typename future_type<
                        typename hpx::util::decay<Executor>::type,
                        typename bulk_async_execute_result<F, S, Ts...>::type
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
                    results.push_back(exec.async_execute(f, elem, ts...));
                }

                return results;
            }

            template <typename Executor, typename F, typename S, typename ... Ts>
            static auto
            call(int, Executor&& exec, F && f, S const& shape, Ts &&... ts)
            ->  decltype(
                    exec.bulk_async_execute(std::forward<F>(f), shape,
                        std::forward<Ts>(ts)...)
                )
            {
                return exec.bulk_async_execute(std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...);
            }
        };

        template <typename Executor, typename F, typename S, typename ... Ts>
        auto call_bulk_async_execute(Executor&& exec, F && f, S const& shape,
                Ts &&... ts)
        ->  decltype(
                bulk_async_execute_helper::call(0, std::forward<Executor>(exec),
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...)
            )
        {
            return bulk_async_execute_helper::call(
                0, std::forward<Executor>(exec), std::forward<F>(f), shape,
                std::forward<Ts>(ts)...);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Shape, bool IsVoid, typename ... Ts>
        struct bulk_execute_result_impl;

        template <typename F, typename Shape, typename ... Ts>
        struct bulk_execute_result_impl<F, Shape, false, Ts...>
        {
            typedef typename hpx::util::detail::unwrap_impl<
                    typename bulk_async_execute_result<F, Shape, Ts...>::type
                >::type type;
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
                    typename bulk_async_execute_result<F, Shape, Ts...>::type
                >::value,
                Ts...>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct bulk_result_helper
        {
            typedef std::vector<T> type;
        };

        template <>
        struct bulk_result_helper<void>
        {
            typedef void type;
        };

        struct bulk_execute_helper
        {
            // returns void if F returns void
            template <typename Executor, typename F, typename S, typename ... Ts>
            static auto call(hpx::traits::detail::wrap_int, Executor&& exec,
                    F && f, S const& shape, Ts &&... ts)
            ->  typename bulk_result_helper<decltype(
                    exec.async_execute(f, *boost::begin(shape), ts...).get()
                )>::type
            {
                std::vector<typename future_type<
                        typename hpx::util::decay<Executor>::type,
                        typename bulk_async_execute_result<F, S, Ts...>::type
                    >::type> results;

// Before Boost V1.56 boost::size() does not respect the iterator category of
// its argument.
#if BOOST_VERSION < 105600
                results.reserve(
                    std::distance(boost::begin(shape), boost::end(shape)));
#else
                results.reserve(boost::size(shape));
#endif

                try {
                    for (auto const& elem: shape)
                    {
                        results.push_back(exec.async_execute(f, elem, ts...));
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

            template <typename Executor, typename F, typename S, typename ... Ts>
            static auto
            call(int, Executor&& exec, F && f, S const& shape, Ts &&... ts)
            ->  decltype(
                    std::declval<Executor>().bulk_execute(std::forward<F>(f), shape,
                        std::forward<Ts>(ts)...)
                )
            {
                return exec.bulk_execute(std::forward<F>(f), shape,
                    std::forward<Ts>(ts)...);
            }
        };

        template <typename Executor, typename F, typename S, typename ... Ts>
        auto call_bulk_execute(Executor&& exec, F && f, S const& shape, Ts &&... ts)
        ->  decltype(
                bulk_execute_helper::call(0, std::forward<Executor>(exec),
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...)
            )
        {
            return bulk_execute_helper::call(0, std::forward<Executor>(exec),
                std::forward<F>(f), shape, std::forward<Ts>(ts)...);
        }
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The executor_traits type is used to request execution agents from an
    /// executor. It is analogous to the interaction between containers and
    /// allocator_traits.
    ///
    /// \note For maximum implementation flexibility, executor_traits does not
    ///       require executors to implement a particular exception reporting
    ///       mechanism. Executors may choose whether or not to report
    ///       exceptions, and if so, in what manner they are communicated back
    ///       to the caller. However, we expect many executors to report
    ///       exceptions in a manner consistent with the behavior of execution
    ///       policies described by the Parallelism TS, where multiple exceptions
    ///       are collected into an exception_list. This list would be reported
    ///       through async_execute()'s returned future, or thrown directly by
    ///       execute().
    ///
    template <typename Executor, typename Enable>
    struct executor_traits
    {
        /// The type of the executor associated with this instance of
        /// \a executor_traits
        typedef Executor executor_type;

        /// The category of agents created by the bulk-form execute() and
        /// async_execute()
        ///
        /// \note This evaluates to executor_type::execution_category if it
        ///       exists; otherwise it evaluates to \a parallel_execution_tag.
        ///
        typedef typename detail::execution_category<executor_type>::type
            execution_category;

        /// The type of future returned by async_execute()
        ///
        /// \note This evaluates to executor_type::future_type<T> if it exists;
        ///       otherwise it evaluates to \a hpx::future<T>
        ///
        template <typename T>
        struct future
        {
            /// The future type returned from async_execute
            typedef typename detail::future_type<executor_type, T>::type type;
        };

        /// \brief Singleton form of asynchronous fire & forget execution agent
        ///        creation.
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
        /// \note This calls exec.apply_execute(f, ts...), if available, otherwise
        ///       it calls exec.async_execute() while discarding the returned
        ///       future
        ///
        template <typename Executor_, typename F, typename ... Ts>
        static void apply_execute(Executor_ && exec, F && f, Ts &&... ts)
        {
            detail::call_apply_execute(std::forward<Executor_>(exec),
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        /// \brief Singleton form of asynchronous execution agent creation.
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
        /// \note This calls exec.async_execute(f)
        ///
        /// \returns f(ts...)'s result through a future
        ///
        template <typename Executor_, typename F, typename ... Ts>
        static auto async_execute(Executor_ && exec, F && f, Ts &&... ts)
        ->  decltype(
                std::declval<Executor_>().async_execute(std::forward<F>(f),
                    std::forward<Ts>(ts)...)
            )
        {
            return exec.async_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        /// \brief Singleton form of synchronous execution agent creation.
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
        /// \note This calls exec.execute(f) if it exists;
        ///       otherwise hpx::async(f).get()
        ///
        template <typename Executor_, typename F, typename ... Ts>
        static auto execute(Executor_ && exec, F && f, Ts &&...ts)
        ->  decltype(
                detail::call_execute(std::forward<Executor_>(exec),
                    std::forward<F>(f), std::forward<Ts>(ts)...)
            )
        {
            return detail::call_execute(std::forward<Executor_>(exec),
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        /// \brief Bulk form of asynchronous execution agent creation
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
        /// \returns The return type of \a executor_type::async_execute if
        ///          defined by \a executor_type. Otherwise a vector
        ///          of futures holding the returned value of each invocation
        ///          of \a f.
        ///
        /// \note This calls exec.async_execute(f, shape) if it exists;
        ///       otherwise it executes hpx::async(f, i) as often as needed.
        ///
        template <typename Executor_, typename F, typename Shape,
            typename ... Ts>
        static auto
        bulk_async_execute(Executor_ && exec, F && f, Shape const& shape,
            Ts &&... ts)
        ->  decltype(
                detail::call_bulk_async_execute(std::forward<Executor_>(exec),
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...)
            )
        {
            return detail::call_bulk_async_execute(
                std::forward<Executor_>(exec), std::forward<F>(f), shape,
                std::forward<Ts>(ts)...);
        }

        /// \brief Bulk form of synchronous execution agent creation
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
        /// \returns The return type of \a executor_type::execute if defined
        ///          by \a executor_type. Otherwise a vector holding the
        ///          returned value of each invocation of \a f except when
        ///          \a f returns void, which case void is returned.
        ///
        /// \note This calls exec.execute(f, shape) if it exists;
        ///       otherwise it executes hpx::async(f, i) as often as needed.
        ///
        template <typename Executor_, typename F, typename Shape,
            typename ... Ts>
        static auto
        bulk_execute(Executor_ && exec, F && f, Shape const& shape, Ts &&... ts)
        ->  decltype(
                detail::call_bulk_execute(std::forward<Executor_>(exec),
                    std::forward<F>(f), shape, std::forward<Ts>(ts)...)
            )
        {
            return detail::call_bulk_execute(std::forward<Executor_>(exec),
                std::forward<F>(f), shape, std::forward<Ts>(ts)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    /// 1. The type is_executor can be used to detect executor types
    ///    for the purpose of excluding function signatures
    ///    from otherwise ambiguous overload resolution participation.
    /// 2. If T is the type of a standard or implementation-defined executor,
    ///    is_executor<T> shall be publicly derived from
    ///    integral_constant<bool, true>, otherwise from
    ///    integral_constant<bool, false>.
    /// 3. The behavior of a program that adds specializations for
    ///    is_executor is undefined.
    ///
    template <typename T>
    struct is_executor;         // defined in hpx/traits/is_executor.hpp
}}}

#endif
