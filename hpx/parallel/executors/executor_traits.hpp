//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/executor_traits.hpp

#if !defined(HPX_PARALLEL_EXECUTOR_TRAITS_MAY_10_2015_1128AM)
#define HPX_PARALLEL_EXECUTOR_TRAITS_MAY_10_2015_1128AM

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/async.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/unwrapped.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>

#include <type_traits>
#include <utility>
#include <stdexcept>

#include <boost/range/functions.hpp>
#include <boost/range/irange.hpp>
#include <boost/throw_exception.hpp>

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION < 40700
#define HPX_ENABLE_WORKAROUND_FOR_GCC46
#endif

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

    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename Category1, typename Category2>
        struct is_not_weaker
          : std::false_type
        {};

        template <typename Category>
        struct is_not_weaker<Category, Category>
          : std::true_type
        {};

        template <>
        struct is_not_weaker<parallel_execution_tag, vector_execution_tag>
          : std::true_type
        {};

        template <>
        struct is_not_weaker<sequential_execution_tag, vector_execution_tag>
          : std::true_type
        {};

        template <>
        struct is_not_weaker<sequential_execution_tag, parallel_execution_tag>
          : std::true_type
        {};
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL

        // wraps int so that int argument is favored over wrap_int
        struct wrap_int
        {
            wrap_int(int) {}
        };

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
                typename Executor::future_type
            >::type>
        {
            typedef typename Executor::future_type type;
        };

        ///////////////////////////////////////////////////////////////////////
        struct apply_helper
        {
            template <typename Executor, typename F>
            static void call(wrap_int, Executor& exec, F && f)
            {
                exec.async_execute(std::forward<F>(f));
            }

            template <typename Executor, typename F>
            static auto call(int, Executor& exec, F && f)
            ->  decltype(exec.apply_execute(std::forward<F>(f)))
            {
                exec.apply_execute(std::forward<F>(f));
            }
        };

        template <typename Executor, typename F>
        void call_apply_execute(Executor& exec, F && f)
        {
            apply_helper::call(0, exec, std::forward<F>(f));
        }

        ///////////////////////////////////////////////////////////////////////
        struct execute_helper
        {
            template <typename Executor, typename F>
            static auto call(wrap_int, Executor& exec, F && f)
            ->  decltype(exec.async_execute(std::forward<F>(f)).get())
            {
                try {
                    return exec.async_execute(std::forward<F>(f)).get();
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

            template <typename Executor, typename F>
            static auto call(int, Executor& exec, F && f)
            ->  decltype(exec.execute(std::forward<F>(f)))
            {
                return exec.execute(std::forward<F>(f));
            }
        };

        template <typename Executor, typename F>
        auto call_execute(Executor& exec, F && f)
#if defined(HPX_ENABLE_WORKAROUND_FOR_GCC46)
        ->  typename hpx::util::result_of<
                typename hpx::util::decay<F>::type()
            >::type
#else
        ->  decltype(execute_helper::call(0, exec, std::forward<F>(f)))
#endif
        {
            return execute_helper::call(0, exec, std::forward<F>(f));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Shape>
        struct bulk_async_execute_result
        {
            typedef typename
                    boost::range_const_iterator<Shape>::type
                iterator_type;
            typedef typename
                    std::iterator_traits<iterator_type>::value_type
                value_type;
            typedef typename hpx::util::result_of<
                    typename hpx::util::decay<F>::type(value_type)
                >::type type;
        };

        ///////////////////////////////////////////////////////////////////////
        struct bulk_async_execute_helper
        {
            template <typename Executor, typename F, typename S>
            static auto call(wrap_int, Executor& exec, F && f, S const& shape)
            ->  std::vector<decltype(
                    exec.async_execute(
                        hpx::util::deferred_call(f, *boost::begin(shape))
                    )
                )>
            {
                std::vector<typename future_type<
                        Executor,
                        typename bulk_async_execute_result<F, S>::type
                    >::type> results;
                results.reserve(boost::size(shape));

                for (auto const& elem: shape)
                {
                    results.push_back(exec.async_execute(
                        hpx::util::deferred_call(f, elem)
                    ));
                }

                return results;
            }

            template <typename Executor, typename F, typename S>
            static auto call(int, Executor& exec, F && f, S const& shape)
            ->  decltype(exec.bulk_async_execute(std::forward<F>(f), shape))
            {
                return exec.bulk_async_execute(std::forward<F>(f), shape);
            }
        };

        template <typename Executor, typename F, typename S>
        auto call_bulk_async_execute(Executor& exec, F && f, S const& shape)
#if defined(HPX_ENABLE_WORKAROUND_FOR_GCC46)
        ->  std::vector<typename future_type<
                Executor, typename bulk_async_execute_result<F, S>::type
            >::type>
#else
        ->  decltype(
                bulk_async_execute_helper::call(0, exec, std::forward<F>(f),
                    shape)
            )
#endif
        {
            return bulk_async_execute_helper::call(
                0, exec, std::forward<F>(f), shape);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename F, typename Shape, typename Enable = void>
        struct bulk_execute_result
        {
            typedef hpx::util::detail::unwrap_impl<
                    typename detail::bulk_async_execute_result<F, Shape>::type
                > type;
        };

        template <typename F, typename Shape>
        struct bulk_execute_result<F, Shape,
            typename boost::enable_if_c<
                boost::is_void<
                    typename detail::bulk_async_execute_result<F, Shape>::type
                >::value
            >::type>
        {
            typedef void type;
        };

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
            template <typename Executor, typename F, typename S>
            static auto call(wrap_int, Executor& exec, F && f, S const& shape)
            ->  typename bulk_result_helper<decltype(
                    exec.async_execute(
                        hpx::util::deferred_call(f, *boost::begin(shape))
                    ).get()
                )>::type
            {
                std::vector<typename future_type<
                        Executor,
                        typename bulk_async_execute_result<F, S>::type
                    >::type> results;
                results.reserve(boost::size(shape));

                try {
                    for (auto const& elem: shape)
                    {
                        results.push_back(
                            exec.async_execute(hpx::util::deferred_call(f, elem))
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

            template <typename Executor, typename F, typename S>
            static auto call(int, Executor& exec, F && f, S const& shape)
            ->  decltype(exec.bulk_execute(std::forward<F>(f), shape))
            {
                return exec.bulk_execute(std::forward<F>(f), shape);
            }
        };

        template <typename Executor, typename F, typename S>
        auto call_bulk_execute(Executor& exec, F && f, S const& shape)
#if defined(HPX_ENABLE_WORKAROUND_FOR_GCC46)
        ->  typename detail::bulk_execute_result<F, S>::type
#else
        ->  decltype(bulk_execute_helper::call(0, exec, std::forward<F>(f), shape))
#endif
        {
            return bulk_execute_helper::call(0, exec, std::forward<F>(f), shape);
        }

        ///////////////////////////////////////////////////////////////////////
        struct os_thread_count_helper
        {
            template <typename Executor>
            static std::size_t call(wrap_int, Executor& exec)
            {
                return hpx::get_os_thread_count();
            }

            template <typename Executor>
            static auto call(int, Executor& exec)
            ->  decltype(exec.os_thread_count())
            {
                return exec.os_thread_count();
            }
        };

        template <typename Executor>
        std::size_t call_os_thread_count(Executor& exec)
        {
            return os_thread_count_helper::call(0, exec);
        }

        ///////////////////////////////////////////////////////////////////////
        struct has_pending_closures_helper
        {
            template <typename Executor>
            static auto call(wrap_int, Executor& exec) -> bool
            {
                return false;   // assume stateless scheduling
            }

            template <typename Executor>
            static auto call(int, Executor& exec)
                ->  decltype(exec.has_pending_closures())
            {
                return exec.has_pending_closures();
            }
        };

        template <typename Executor>
        bool call_has_pending_closures(Executor& exec)
        {
            return has_pending_closures_helper::call(0, exec);
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
        ///
        /// \note This calls exec.apply_execute(f), if available, otherwise
        ///       it calls exec.async_execute() while discarding the returned
        ///       future
        ///
        template <typename F>
        static void apply_execute(executor_type& exec, F && f)
        {
            detail::call_apply_execute(exec, std::forward<F>(f));
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
        ///
        /// \note Executors have to implement only `async_execute()`. All other
        ///       functions will be emulated by this `executor_traits` in terms
        ///       of this single basic primitive. However, some executors will
        ///       naturally specialize all operations for maximum efficiency.
        ///
        /// \note This calls exec.async_execute(f)
        ///
        /// \returns f()'s result through a future
        ///
        template <typename F>
        static auto async_execute(executor_type& exec, F && f)
#if defined(HPX_ENABLE_WORKAROUND_FOR_GCC46)
        ->  typename future<
                typename hpx::util::result_of<
                    typename hpx::util::decay<F>::type()
                >::type
            >::type
#else
        ->  decltype(exec.async_execute(std::forward<F>(f)))
#endif
        {
            return exec.async_execute(std::forward<F>(f));
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
        /// \returns f()'s result
        ///
        /// \note This calls exec.execute(f) if it exists;
        ///       otherwise hpx::async(f).get()
        ///
        template <typename F>
        static auto execute(executor_type& exec, F && f)
#if defined(HPX_ENABLE_WORKAROUND_FOR_GCC46)
        ->  typename hpx::util::result_of<
                typename hpx::util::decay<F>::type()
            >::type
#else
        ->  decltype(detail::call_execute(exec, std::forward<F>(f)))
#endif
        {
            return detail::call_execute(exec, std::forward<F>(f));
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
        /// \returns The return type of \a executor_type::async_execute if
        ///          defined by \a executor_type. Otherwise a vector
        ///          of futures holding the returned value of each invocation
        ///          of \a f.
        ///
        /// \note This calls exec.async_execute(f, shape) if it exists;
        ///       otherwise it executes hpx::async(f, i) as often as needed.
        ///
        template <typename F, typename Shape>
        static auto
        async_execute(executor_type& exec, F && f, Shape const& shape)
#if defined(HPX_ENABLE_WORKAROUND_FOR_GCC46)
        ->  std::vector<typename future<
                typename detail::bulk_async_execute_result<F, Shape>::type
            >::type>
#else
        ->  decltype(
                detail::call_bulk_async_execute(exec, std::forward<F>(f), shape)
            )
#endif
        {
            return detail::call_bulk_async_execute(
                exec, std::forward<F>(f), shape);
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
        /// \returns The return type of \a executor_type::execute if defined
        ///          by \a executor_type. Otherwise a vector holding the
        ///          returned value of each invocation of \a f except when
        ///          \a f returns void, which case void is returned.
        ///
        /// \note This calls exec.execute(f, shape) if it exists;
        ///       otherwise it executes hpx::async(f, i) as often as needed.
        ///
        template <typename F, typename Shape>
        static auto execute(executor_type& exec, F && f, Shape const& shape)
#if defined(HPX_ENABLE_WORKAROUND_FOR_GCC46)
        ->  typename detail::bulk_execute_result<F, Shape>::type
#else
        ->  decltype(detail::call_bulk_execute(exec, std::forward<F>(f), shape))
#endif
        {
            return detail::call_bulk_execute(exec, std::forward<F>(f), shape);
        }

        /// Retrieve the number of (kernel-)threads used by the associated
        /// executor.
        ///
        /// \param exec  [in] The executor object to use for scheduling of the
        ///              function \a f.
        ///
        /// \note This calls exec.os_thread_count() if it exists;
        ///       otherwise it executes hpx::get_os_thread_count().
        ///
        static std::size_t os_thread_count(executor_type const& exec)
        {
            return detail::call_os_thread_count(exec);
        }

        /// Retrieve whether this executor has operations pending or not.
        ///
        /// \param exec  [in] The executor object to use for scheduling of the
        ///              function \a f.
        ///
        /// \note If the executor does not expose this information, this call
        ///       will always return \a false
        ///
        static bool has_pending_closures(executor_type& exec)
        {
            return detail::call_has_pending_closures(exec);
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

#undef HPX_ENABLE_WORKAROUND_FOR_GCC46

#endif
