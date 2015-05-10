//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executor_traits.hpp

#if !defined(HPX_PARALLEL_EXECUTOR_TRAITS_MAY_10_2015_1128AM)
#define HPX_PARALLEL_EXECUTOR_TRAITS_MAY_10_2015_1128AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/result_of.hpp>
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

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename Enable = void>
        struct execution_category
        {
            typedef parallel_execution_tag type;
        };

        template <typename Executor>
        struct execution_category<
            Executor,
            typename util::always_void<
                typename Executor::execution_category
            >::type>
        {
            typedef typename Executor::execution_category type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename Enable = void>
        struct shape_type
        {
            typedef boost::integer_range<std::size_t> type;
        };

        template <typename Executor>
        struct shape_type<
            Executor,
            typename util::always_void<
                typename Executor::shape_type
            >::type>
        {
            typedef typename Executor::shape_type type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename T, typename Enable = void>
        struct future_type
        {
            typedef hpx::future<T> type;
        };

        template <typename Executor, typename T>
        struct future_type<
            Executor, T,
            typename util::always_void<
                typename Executor::future_type
            >::type>
        {
            typedef typename Executor::future_type type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename F, typename Enable = void>
        struct async_execute
        {
            typedef typename util::result_of<F()>::type result_type;
            typedef typename future_type<Executor, result_type>::type type;

            static type call(Executor&, F const& f)
            {
                return hpx::async(f);
            }
        };

        template <typename Executor, typename F>
        struct async_execute<
            Executor, F,
            typename util::always_void<
                decltype(
                    std::declval<Executor>().async_execute(std::declval<F>())
                )
            >::type>
        {
            typedef typename util::result_of<F()>::type result_type;
            typedef typename future_type<Executor, result_type>::type type;

            static type call(Executor& exec, F const& f)
            {
                return exec.async_execute(f);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename F, typename Enable = void>
        struct execute
        {
            typedef typename util::result_of<F()>::type type;

            static type call(Executor&, F const& f)
            {
                return hpx::async(f).get();
            }
        };

        template <typename Executor, typename F>
        struct execute<
            Executor, F,
            typename util::always_void<
                decltype(std::declval<Executor>().execute(std::declval<F>()))
            >::type>
        {
            typedef typename util::result_of<F()>::type type;

            static type call(Executor& exec, F const& f)
            {
                return exec.execute(f);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename F, typename S,
            typename Enable = void>
        struct bulk_async_execute
        {
            typedef typename future_type<Executor, void>::type type;

            static type call(Executor&, F const& f, S shape)
            {
                std::vector<hpx::future<void> > results;

                auto end = boost::end(shape);
                for (auto it = boost::begin(shape); it != end; ++it)
                    results.push_back(hpx::async(f, *it));

                return hpx::when_all(results);
            }
        };

        template <typename Executor, typename F, typename S>
        struct bulk_async_execute<
            Executor, F, S,
            typename util::always_void<
                decltype(std::declval<Executor>()
                    .bulk_async_execute(std::declval<F>(), std::declval<S>())
                )
            >::type>
        {
            typedef typename future_type<Executor, void>::type type;

            static type call(Executor& exec, F const& f, S const& shape)
            {
                return exec.bulk_async_execute(f, shape);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename F, typename S,
            typename Enable = void>
        struct bulk_execute
        {
            static void call(Executor&, F const& f, shape_type shape)
            {
                std::vector<hpx::future<void> > results;

                auto end = boost::end(shape);
                for (auto it = boost::begin(shape); it != end; ++it)
                    results.push_back(hpx::async(f, *it));

                hpx::when_all(results).get();
            }
        };

        template <typename Executor, typename F, typename S>
        struct bulk_execute<
            Executor, F, S,
            typename util::always_void<
                decltype(std::declval<Executor>()
                    .bulk_execute(std::declval<F>(), std::declval<S>())
                )
            >::type>
        {
            static void call(Executor& exec, F const& f, S const& shape)
            {
                exec.bulk_execute(f, shape);
            }
        };
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

        /// The type of bulk-form execute() & async_execute()'s shape parameter
        ///
        /// \note This evaluates to executor_type::shape_type if it exists;
        ///       otherwise this evaluates to a range which iterates over
        ///       std::size_t.
        ///
        typedef typename detail::shape_type<executor_type>::type shape_type;

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
        template <typename F>
        static future<void>
        async_execute(executor_type& exec, F f, shape_type shape)
        {
            return detail::bulk_async_execute<executor_type, F, shape_type>::
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
        template <typename F>
        static void execute(executor_type& exec, F f, shape_type shape)
        {
            return detail::bulk_execute<executor_type, F, shape_type>::
                call(exec, f, shape);
        }
    };
}}}

#endif
