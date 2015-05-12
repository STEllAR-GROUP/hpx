//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/executor_traits.hpp

#if !defined(HPX_PARALLEL_EXECUTOR_TRAITS_MAY_10_2015_1128AM)
#define HPX_PARALLEL_EXECUTOR_TRAITS_MAY_10_2015_1128AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>

#include <type_traits>

#include <boost/range/functions.hpp>
#include <boost/range/irange.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    /// Function invocations executed by a group of sequential execution agents
    /// execute in sequential order.
    struct sequential_execution_tag {};

    /// Function invocations executed by a group of parallel execution agents
    /// execute in unordered fashion. Any such invocations executing in the
    /// same thread are indeterminately sequenced with respect to each other.
    ///
    /// \note \a parallel_execution_tag is weaker than
    ///       \a sequential_execution_tag.
    struct parallel_execution_tag {};

    /// Function invocations executed by a group of vector execution agents are
    /// permitted to execute in unordered fashion when executed in different
    /// threads, and un-sequenced with respect to one another when executed in
    /// the same thread.
    ///
    /// \note \a vector_execution_tag is weaker than
    ///       \a parallel_execution_tag.
    struct vector_execution_tag {};

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
#if defined(BOOST_NO_SFINAE_EXPR) || defined(BOOST_NO_CXX11_DECLTYPE_N3276)
        template <typename T, typename NameGetter>
        struct has_member_impl
        {
            typedef char yes;
            typedef long no;

            template <typename C>
            static yes f(typename NameGetter::template get<C>*);

            template <typename C>
            static no f(...);

            static const bool value = (sizeof(f<T>(0)) == sizeof(yes));
        };

        template <typename T, typename NameGetter>
        struct has_member
          : std::integral_constant<bool, has_member_impl<T, NameGetter>::value>
        {};
#endif

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename Enable = void>
        struct execution_category
        {
            typedef parallel_execution_tag type;
        };

        template <typename Executor>
        struct execution_category<Executor,
            typename util::always_void<
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
            typename util::always_void<typename Executor::future_type>::type>
        {
            typedef typename Executor::future_type type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename F, typename Enable = void>
        struct async_execute
        {
            typedef typename util::result_of<F()>::type result_type;
            typedef typename future_type<Executor, result_type>::type type;

            template <typename ...Ts>
            static type call(Executor&, F const& f, Ts&&... ts)
            {
                return hpx::async(f, std::forward<Ts>(ts)...);
            }
        };

#if defined(BOOST_NO_SFINAE_EXPR) || defined(BOOST_NO_CXX11_DECLTYPE_N3276)
        template <typename Executor, typename F>
        struct check_has_async_execute
        {
            typedef typename util::result_of<F()>::type result_type;
            typedef typename future_type<Executor, result_type>::type type;

            template <typename T, type (T::*)(F) = &T::async_execute>
            struct get {};
        };

        template <typename Executor, typename F>
        struct async_execute<Executor, F,
            typename std::enable_if<
                has_member<Executor, check_has_async_execute<
                    Executor, F
                > >::value
            >::type>
        {
            typedef typename util::result_of<F()>::type result_type;
            typedef typename future_type<Executor, result_type>::type type;

            template <typename ...Ts>
            static type call(Executor& exec, F const& f, Ts&&... ts)
            {
                return exec.async_execute(f, std::forward<Ts>(ts)...);
            }
        };
#else
        template <typename Executor, typename F>
        struct async_execute<Executor, F,
            typename util::always_void<decltype(
                std::declval<Executor>().async_execute(std::declval<F>())
            )>::type>
        {
            typedef typename util::result_of<F()>::type result_type;
            typedef typename future_type<Executor, result_type>::type type;

            template <typename ...Ts>
            static type call(Executor& exec, F const& f, Ts&&... ts)
            {
                return exec.async_execute(f, std::forward<Ts>(ts)...);
            }
        };
#endif

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename F, typename Enable = void>
        struct execute
        {
            typedef typename util::result_of<F()>::type type;

            template <typename ...Ts>
            static type call(Executor&, F const& f, Ts&&... ts)
            {
                return hpx::async(f, std::forward<Ts>(ts)...).get();
            }
        };

#if defined(BOOST_NO_SFINAE_EXPR) || defined(BOOST_NO_CXX11_DECLTYPE_N3276)
        template <typename Executor, typename F>
        struct check_has_execute
        {
            typedef typename util::result_of<F()>::type type;

            template <typename T, type (T::*)(F) = &T::execute>
            struct get {};
        };

        template <typename Executor, typename F>
        struct execute<Executor, F,
            typename std::enable_if<
                has_member<Executor, check_has_execute<
                    Executor, F
                > >::value
            >::type>
        {
            typedef typename util::result_of<F()>::type type;

            template <typename ...Ts>
            static type call(Executor& exec, F const& f, Ts&&... ts)
            {
                return exec.execute(f, std::forward<Ts>(ts)...);
            }
        };
#else
        template <typename Executor, typename F>
        struct execute<Executor, F,
            typename util::always_void<
                decltype(std::declval<Executor>().execute(std::declval<F>()))
            >::type>
        {
            typedef typename util::result_of<F()>::type type;

            template <typename ...Ts>
            static type call(Executor& exec, F const& f, Ts&&... ts)
            {
                return exec.execute(f, std::forward<Ts>(ts)...);
            }
        };
#endif

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename F, typename S,
            typename Enable = void>
        struct bulk_async_execute
        {
            typedef typename future_type<Executor, void>::type type;

            static type call(Executor& exec, F const& f, S const& shape)
            {
                std::vector<hpx::future<void> > results;
                for (auto const& elem: shape)
                {
                    results.push_back(
                        async_execute<Executor, F>::call(exec, f, elem)
                    );
                }
                return hpx::when_all(results);
            }
        };

#if defined(BOOST_NO_SFINAE_EXPR) || defined(BOOST_NO_CXX11_DECLTYPE_N3276)
        template <typename Executor, typename F, typename S>
        struct check_has_bulk_async_execute
        {
            typedef typename future_type<Executor, void>::type type;

            template <typename T,
                type (T::*)(F, S const&) = &T::bulk_async_execute>
            struct get {};
        };

        template <typename Executor, typename F, typename S>
        struct bulk_async_execute<Executor, F, S,
            typename std::enable_if<
                has_member<Executor, check_has_bulk_async_execute<
                    Executor, F, S
                > >::value
            >::type>
        {
            typedef typename future_type<Executor, void>::type type;

            static type call(Executor& exec, F const& f, S const& shape)
            {
                return exec.bulk_async_execute(f, shape);
            }
        };
#else
        template <typename Executor, typename F, typename S>
        struct bulk_async_execute<Executor, F, S,
            typename util::always_void<decltype(
                declval<Executor>()
                    .bulk_async_execute(std::declval<F>(), std::declval<S>())
            )>::type>
        {
            typedef typename future_type<Executor, void>::type type;

            static type call(Executor& exec, F const& f, S const& shape)
            {
                return exec.bulk_async_execute(f, shape);
            }
        };
#endif

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename F, typename S,
            typename Enable = void>
        struct bulk_execute
        {
            static void call(Executor& exec, F const& f, S const& shape)
            {
                std::vector<hpx::future<void> > results;
                for (auto const& elem: shape)
                {
                    results.push_back(
                        async_execute<Executor, F>::call(exec, f, elem)
                    );
                }
                hpx::when_all(results).get();
            }
        };

#if defined(BOOST_NO_SFINAE_EXPR) || defined(BOOST_NO_CXX11_DECLTYPE_N3276)
        template <typename F, typename S>
        struct check_has_bulk_execute
        {
            template <typename T, void (T::*)(F, S const&) = &T::bulk_execute>
            struct get {};
        };

        template <typename Executor, typename F, typename S>
        struct bulk_execute<Executor, F, S,
            typename std::enable_if<
                has_member<Executor, check_has_bulk_execute<F, S> >::value
            >::type>
        {
            static void call(Executor& exec, F const& f, S const& shape)
            {
                exec.bulk_execute(f, shape);
            }
        };
#else
        template <typename Executor, typename F, typename S>
        struct bulk_execute<Executor, F, S,
            typename util::always_void<decltype(
                std::declval<Executor>()
                    .bulk_execute(std::declval<F>(), std::declval<S>())
            )>::type>
        {
            static void call(Executor& exec, F const& f, S const& shape)
            {
                exec.bulk_execute(f, shape);
            }
        };
#endif
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor>
    class executor_traits
    {
    public:
        /// The type of the executor
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
            typedef typename detail::future_type<executor_type, T>::type type;
        };

        /// brief Singleton form of asynchronous execution agent creation.
        ///
        /// This asynchronously creates a single function invocation f() using
        /// the associated executor.
        ///
        /// \param exec [in] The executor object to use for scheduling of the
        ///             function \a f.
        /// \param f    [in] The function which will be scheduled using the
        ///             given executor.
        ///
        /// \returns f()'s result through a future
        ///
        /// \note This calls exec.async_execute(f) if it exists;
        ///       otherwise hpx::async(f)
        ///
        template <typename F>
        static typename detail::async_execute<executor_type, F>::type
        async_execute(executor_type& exec, F f)
        {
            return detail::async_execute<executor_type, F>::call(exec, f);
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
        ///
        /// \returns f()'s result through a future
        ///
        /// \note This calls exec.execute(f) if it exists;
        ///       otherwise hpx::async(f).get()
        ///
        template <typename F>
        static typename detail::execute<executor_type, F>::type
        execute(executor_type& exec, F f)
        {
            return detail::execute<executor_type, F>::call(exec, f);
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
        ///
        /// \returns A future object representing which becomes ready once all
        ///          scheduled functions have finished executing.
        ///
        /// \note This calls exec.async_execute(f, shape) if it exists;
        ///       otherwise it executes hpx::async(f, i) as often as needed.
        ///
        template <typename F, typename Shape>
        static future<void>
        async_execute(executor_type& exec, F const& f, Shape const& shape)
        {
            return detail::bulk_async_execute<executor_type, F, Shape>::
                call(exec, f, shape);
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
        ///
        /// \note This calls exec.execute(f, shape) if it exists;
        ///       otherwise it executes hpx::async(f, i) as often as needed.
        ///
        template <typename F, typename Shape>
        static void execute(executor_type& exec, F const& f, Shape const& shape)
        {
            return detail::bulk_execute<executor_type, F, Shape>::
                call(exec, f, shape);
        }
    };
}}}

#endif
