//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nounnamed

#if !defined(HPX_PARALLEL_EXECUTORS_EXECUTION_PARAMETER_FWD_MAY_19_2017_0120PM)
#define HPX_PARALLEL_EXECUTORS_EXECUTION_PARAMETER_FWD_MAY_19_2017_0120PM

#include <hpx/config.hpp>
#include <hpx/parallel/executors/execution_fwd.hpp>
#include <hpx/traits/executor_traits.hpp>
#include <hpx/util/decay.hpp>

#include <cstddef>
#include <utility>

namespace hpx { namespace parallel { namespace execution
{
    ///////////////////////////////////////////////////////////////////////////
    // Define infrastructure for customization points
    namespace detail
    {
        /// \cond NOINTERNAL
        struct get_chunk_size_tag {};
        struct maximal_number_of_chunks_tag {};
        struct reset_thread_distribution_tag {};
        struct count_processing_units_tag {};
        struct mark_begin_execution_tag {};
        struct mark_end_execution_tag {};

#if defined(HPX_HAVE_CXX14_RETURN_TYPE_DEDUCTION)
        // forward declare customization point implementations
        template <>
        struct customization_point<get_chunk_size_tag>
        {
            template <typename Parameters, typename Executor, typename F>
            HPX_FORCEINLINE auto operator()(Parameters&& params,
                Executor&& exec, F&& f, std::size_t cores,
                std::size_t num_tasks) const;
        };

        template <>
        struct customization_point<maximal_number_of_chunks_tag>
        {
            template <typename Parameters, typename Executor>
            HPX_FORCEINLINE auto operator()(Parameters&& params,
                Executor&& exec, std::size_t cores,
                std::size_t num_tasks) const;
        };

        template <>
        struct customization_point<reset_thread_distribution_tag>
        {
            template <typename Parameters, typename Executor>
            HPX_FORCEINLINE auto operator()(
                Parameters&& params, Executor&& exec) const;
        };

        template <>
        struct customization_point<count_processing_units_tag>
        {
            template <typename Parameters, typename Executor>
            HPX_FORCEINLINE auto operator()(Parameters&& params,
                Executor&& exec) const;
        };

        template <>
        struct customization_point<mark_begin_execution_tag>
        {
            template <typename Parameters, typename Executor>
            HPX_FORCEINLINE auto operator()(Parameters&& params,
                Executor&& exec) const;
        };

        template <>
        struct customization_point<mark_end_execution_tag>
        {
            template <typename Parameters, typename Executor>
            HPX_FORCEINLINE auto operator()(Parameters&& params,
                Executor&& exec) const;
        };
#endif
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    // Executor information customization points
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename Parameters, typename Executor, typename Enable = void>
        struct get_chunk_size_fn_helper;

        template <typename Parameters, typename Executor, typename Enable = void>
        struct maximal_number_of_chunks_fn_helper;

        template <typename Parameters, typename Executor, typename Enable = void>
        struct reset_thread_distribution_fn_helper;

        template <typename Parameters, typename Executor, typename Enable = void>
        struct count_processing_units_fn_helper;

        template <typename Parameters, typename Executor, typename Enable = void>
        struct mark_begin_execution_fn_helper;

        template <typename Parameters, typename Executor, typename Enable = void>
        struct mark_end_execution_fn_helper;
        /// \endcond
    }

    namespace detail
    {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        // get_chunk_size dispatch point
        template <typename Parameters, typename Executor, typename F>
        HPX_FORCEINLINE auto get_chunk_size(Parameters&& params,
            Executor&& exec, F&& f, std::size_t cores, std::size_t num_tasks)
        -> decltype(get_chunk_size_fn_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type,
                    typename hpx::util::decay<Executor>::type
                >::call(std::forward<Parameters>(params),
                    std::forward<Executor>(exec), std::forward<F>(f), cores,
                    num_tasks))
        {
            return get_chunk_size_fn_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type,
                    typename hpx::util::decay<Executor>::type
                >::call(std::forward<Parameters>(params),
                    std::forward<Executor>(exec), std::forward<F>(f), cores,
                    num_tasks);
        }

#if defined(HPX_HAVE_CXX14_RETURN_TYPE_DEDUCTION)
        template <typename Parameters, typename Executor, typename F>
        HPX_FORCEINLINE auto customization_point<get_chunk_size_tag>::
        operator()(Parameters&& params, Executor&& exec, F&& f,
            std::size_t cores, std::size_t num_tasks) const
        {
            return get_chunk_size(std::forward<Parameters>(params),
                std::forward<Executor>(exec), std::forward<F>(f), cores,
                num_tasks);
        }
#else
        template <>
        struct customization_point<get_chunk_size_tag>
        {
        public:
            template <typename Executor, typename Parameters, typename F>
            HPX_FORCEINLINE auto operator()(Parameters&& params,
                    Executor&& exec, F&& f, std::size_t cores,
                    std::size_t num_tasks) const
            -> decltype(get_chunk_size(std::forward<Parameters>(params),
                    std::forward<Executor>(exec), std::forward<F>(f), cores,
                    num_tasks))
            {
                return get_chunk_size(std::forward<Parameters>(params),
                    std::forward<Executor>(exec), std::forward<F>(f), cores,
                    num_tasks);
            }
        };
#endif

        ///////////////////////////////////////////////////////////////////////
        // maximal_number_of_chunks dispatch point
        template <typename Parameters, typename Executor>
        HPX_FORCEINLINE auto maximal_number_of_chunks(Parameters&& params,
                Executor&& exec, std::size_t cores, std::size_t num_tasks)
        ->  decltype(maximal_number_of_chunks_fn_helper<
                        typename hpx::util::decay_unwrap<Parameters>::type,
                        typename hpx::util::decay<Executor>::type
                    >::call(std::forward<Parameters>(params),
                        std::forward<Executor>(exec), cores, num_tasks))
        {
            return maximal_number_of_chunks_fn_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type,
                    typename hpx::util::decay<Executor>::type
                >::call(std::forward<Parameters>(params),
                    std::forward<Executor>(exec), cores, num_tasks);
        }

#if defined(HPX_HAVE_CXX14_RETURN_TYPE_DEDUCTION)
        template <typename Parameters, typename Executor>
        HPX_FORCEINLINE auto customization_point<maximal_number_of_chunks_tag>::
        operator()(Parameters&& params, Executor&& exec, std::size_t cores,
            std::size_t num_tasks) const
        {
            return maximal_number_of_chunks(std::forward<Parameters>(params),
                std::forward<Executor>(exec), cores, num_tasks);
        }
#else
        template <>
        struct customization_point<maximal_number_of_chunks_tag>
        {
        public:
            template <typename Parameters, typename Executor>
            HPX_FORCEINLINE auto operator()(Parameters&& params,
                    Executor&& exec, std::size_t cores,
                    std::size_t num_tasks) const
            ->  decltype(maximal_number_of_chunks(
                    std::forward<Parameters>(params),
                    std::forward<Executor>(exec), cores, num_tasks))
            {
                return maximal_number_of_chunks(
                    std::forward<Parameters>(params),
                    std::forward<Executor>(exec), cores, num_tasks);
            }
        };
#endif

        ///////////////////////////////////////////////////////////////////////
        // reset_thread_distribution dispatch point
        template <typename Parameters, typename Executor>
        HPX_FORCEINLINE auto
        reset_thread_distribution(Parameters&& params, Executor&& exec)
        -> decltype(reset_thread_distribution_fn_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type,
                    typename hpx::util::decay<Executor>::type
                >::call(std::forward<Parameters>(params),
                        std::forward<Executor>(exec)))
        {
            return reset_thread_distribution_fn_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type,
                    typename hpx::util::decay<Executor>::type
                >::call(std::forward<Parameters>(params),
                        std::forward<Executor>(exec));
        }

#if defined(HPX_HAVE_CXX14_RETURN_TYPE_DEDUCTION)
        template <typename Parameters, typename Executor>
        HPX_FORCEINLINE auto customization_point<reset_thread_distribution_tag>::
        operator()(Parameters&& params, Executor&& exec) const
        {
            return reset_thread_distribution(std::forward<Parameters>(params),
                std::forward<Executor>(exec));
        }
#else
        template <>
        struct customization_point<reset_thread_distribution_tag>
        {
        public:
            template <typename Parameters, typename Executor>
            HPX_FORCEINLINE auto operator()(Parameters&& params,
                    Executor&& exec) const
            -> decltype(reset_thread_distribution(
                    std::forward<Parameters>(params),
                    std::forward<Executor>(exec)))
            {
                return reset_thread_distribution(
                    std::forward<Parameters>(params),
                    std::forward<Executor>(exec));
            }
        };
#endif

        ///////////////////////////////////////////////////////////////////////
        // count_processing_units dispatch point
        template <typename Parameters, typename Executor>
        HPX_FORCEINLINE auto
        count_processing_units(Parameters&& params, Executor&& exec)
        -> decltype(count_processing_units_fn_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type,
                    typename hpx::util::decay<Executor>::type
                >::call(std::forward<Parameters>(params),
                    std::forward<Executor>(exec)))
        {
            return count_processing_units_fn_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type,
                    typename hpx::util::decay<Executor>::type
                >::call(
                    std::forward<Parameters>(params),
                    std::forward<Executor>(exec));
        }

#if defined(HPX_HAVE_CXX14_RETURN_TYPE_DEDUCTION)
        template <typename Parameters, typename Executor>
        HPX_FORCEINLINE auto customization_point<count_processing_units_tag>::
        operator()(Parameters&& params, Executor&& exec) const
        {
            return count_processing_units(std::forward<Parameters>(params),
                std::forward<Executor>(exec));
        }
#else
        template <>
        struct customization_point<count_processing_units_tag>
        {
        public:
            template <typename Parameters, typename Executor>
            HPX_FORCEINLINE auto operator()(
                    Parameters&& params, Executor&& exec) const
            -> decltype(count_processing_units(
                    std::forward<Parameters>(params),
                    std::forward<Executor>(exec)))
            {
                return count_processing_units(
                    std::forward<Parameters>(params),
                    std::forward<Executor>(exec));
            }
        };
#endif

        ///////////////////////////////////////////////////////////////////////
        // mark_begin_execution dispatch point
        template <typename Parameters, typename Executor>
        HPX_FORCEINLINE auto mark_begin_execution(Parameters&& params,
                Executor&& exec)
        -> decltype(mark_begin_execution_fn_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type,
                    typename hpx::util::decay<Executor>::type
                >::call(std::forward<Parameters>(params),
                    std::forward<Executor>(exec)))
        {
            return mark_begin_execution_fn_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type,
                    typename hpx::util::decay<Executor>::type
                >::call(std::forward<Parameters>(params),
                    std::forward<Executor>(exec));
        }

#if defined(HPX_HAVE_CXX14_RETURN_TYPE_DEDUCTION)
        template <typename Parameters, typename Executor>
        HPX_FORCEINLINE auto customization_point<mark_begin_execution_tag>::
        operator()(Parameters&& params, Executor&& exec) const
        {
            return mark_begin_execution(std::forward<Parameters>(params),
                std::forward<Executor>(exec));
        }
#else
        template <>
        struct customization_point<mark_begin_execution_tag>
        {
        public:
            template <typename Parameters, typename Executor>
            HPX_FORCEINLINE auto operator()(Parameters&& params,
                    Executor&& exec) const
            -> decltype(mark_begin_execution(std::forward<Parameters>(params),
                    std::forward<Executor>(exec)))
            {
                return mark_begin_execution(std::forward<Parameters>(params),
                    std::forward<Executor>(exec));
            }
        };
#endif

        ///////////////////////////////////////////////////////////////////////
        // mark_end_execution dispatch point
        template <typename Parameters, typename Executor>
        HPX_FORCEINLINE auto mark_end_execution(Parameters&& params,
                Executor&& exec)
        -> decltype(mark_end_execution_fn_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type,
                    typename hpx::util::decay<Executor>::type
                >::call(std::forward<Parameters>(params),
                    std::forward<Executor>(exec)))
        {
            return mark_end_execution_fn_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type,
                    typename hpx::util::decay<Executor>::type
                >::call(std::forward<Parameters>(params),
                    std::forward<Executor>(exec));
        }

#if defined(HPX_HAVE_CXX14_RETURN_TYPE_DEDUCTION)
        template <typename Parameters, typename Executor>
        HPX_FORCEINLINE auto customization_point<mark_end_execution_tag>::
        operator()(Parameters&& params, Executor&& exec) const
        {
            return mark_end_execution(std::forward<Parameters>(params),
                std::forward<Executor>(exec));
        }
#else
        template <>
        struct customization_point<mark_end_execution_tag>
        {
        public:
            template <typename Parameters, typename Executor>
            HPX_FORCEINLINE auto operator()(Parameters&& params,
                    Executor&& exec) const
            -> decltype(mark_end_execution(std::forward<Parameters>(params),
                    std::forward<Executor>(exec)))
            {
                return mark_end_execution(std::forward<Parameters>(params),
                    std::forward<Executor>(exec));
            }
        };
#endif

        /// \endcond
    }

    // define customization points
    namespace
    {
        /// Return the number of invocations of the given function \a f which
        /// should be combined into a single task
        ///
        /// \param params   [in] The executor parameters object to use for
        ///                 determining the chunk size for the given number of
        ///                 tasks \a num_tasks.
        /// \param exec     [in] The executor object which will be used
        ///                 for scheduling of the loop iterations.
        /// \param f        [in] The function which will be optionally scheduled
        ///                 using the given executor.
        /// \param cores    [in] The number of cores the number of chunks
        ///                 should be determined for.
        /// \param num_tasks [in] The number of tasks the chunk size should be
        ///                 determined for
        ///
        /// \note  The parameter \a f is expected to be a nullary function
        ///        returning a `std::size_t` representing the number of
        ///        iteration the function has already executed (i.e. which
        ///        don't have to be scheduled anymore).
        ///
        constexpr detail::customization_point<
                detail::get_chunk_size_tag
            > const& get_chunk_size =
                detail::static_const<detail::customization_point<
                    detail::get_chunk_size_tag
                > >::value;

        /// Return the largest reasonable number of chunks to create for a
        /// single algorithm invocation.
        ///
        /// \param params   [in] The executor parameters object to use for
        ///                 determining the number of chunks for the given
        ///                 number of \a cores.
        /// \param exec     [in] The executor object which will be used
        ///                 for scheduling of the loop iterations.
        /// \param cores    [in] The number of cores the number of chunks
        ///                 should be determined for.
        /// \param num_tasks [in] The number of tasks the chunk size should be
        ///                 determined for
        ///
        constexpr detail::customization_point<
                detail::maximal_number_of_chunks_tag
            > const& maximal_number_of_chunks =
                detail::static_const<detail::customization_point<
                    detail::maximal_number_of_chunks_tag
                > >::value;

        /// Reset the internal round robin thread distribution scheme for the
        /// given executor.
        ///
        /// \param params   [in] The executor parameters object to use for
        ///                 resetting the thread distribution scheme.
        /// \param exec     [in] The executor object to use.
        ///
        /// \note This calls params.reset_thread_distribution(exec) if it exists;
        ///       otherwise it does nothing.
        ///
        constexpr detail::customization_point<
                detail::reset_thread_distribution_tag
            > const& reset_thread_distribution =
                detail::static_const<detail::customization_point<
                    detail::reset_thread_distribution_tag
                > >::value;

        /// Retrieve the number of (kernel-)threads used by the associated
        /// executor.
        ///
        /// \param params [in] The executor parameters object to use as a
        ///              fallback if the executor does not expose
        ///
        /// \note This calls params.processing_units_count() if it exists;
        ///       otherwise it forwards the request to the executor parameters
        ///       object.
        ///
        constexpr detail::customization_point<
                detail::count_processing_units_tag
            > const& count_processing_units =
                detail::static_const<detail::customization_point<
                    detail::count_processing_units_tag
                > >::value;

        /// Mark the begin of a parallel algorithm execution
        ///
        /// \param params [in] The executor parameters object to use as a
        ///              fallback if the executor does not expose
        ///
        /// \note This calls params.mark_begin_execution(exec) if it exists;
        ///       otherwise it does nothing.
        ///
        constexpr detail::customization_point<
                detail::mark_begin_execution_tag
            > const& mark_begin_execution =
                detail::static_const<detail::customization_point<
                    detail::mark_begin_execution_tag
                > >::value;

        /// Mark the end of a parallel algorithm execution
        ///
        /// \param params [in] The executor parameters object to use as a
        ///              fallback if the executor does not expose
        ///
        /// \note This calls params.mark_end_execution(exec) if it exists;
        ///       otherwise it does nothing.
        ///
        constexpr detail::customization_point<
                detail::mark_end_execution_tag
            > const& mark_end_execution =
                detail::static_const<detail::customization_point<
                    detail::mark_end_execution_tag
                > >::value;
    }
}}}

#endif

