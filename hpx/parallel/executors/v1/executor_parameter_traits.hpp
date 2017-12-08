//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/v1/executor_parameter_traits.hpp

#if !defined(HPX_PARALLEL_EXECUTOR_PARAMETER_TRAITS_JUL_30_2015_0914PM)
#define HPX_PARALLEL_EXECUTOR_PARAMETER_TRAITS_JUL_30_2015_0914PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
#include <hpx/lcos/future.hpp>
#include <hpx/traits/detail/wrap_int.hpp>
#include <hpx/traits/v1/is_executor.hpp>
#include <hpx/traits/v1/is_executor_parameters.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/decay.hpp>

#include <hpx/parallel/executors/execution.hpp>
#include <hpx/parallel/executors/execution_parameters.hpp>
#include <hpx/parallel/executors/v1/executor_traits.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v3
{
    ///////////////////////////////////////////////////////////////////////////
    // Placeholder type to use predefined executor parameters
    using sequential_executor_parameters =
        execution::sequential_executor_parameters;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL

        // If an executor exposes 'executor_parameter_type' this type is
        // assumed to represent the default parameters for the given executor
        // type.
        template <typename Executor>
        using extract_executor_parameters =
            execution::extract_executor_parameters<Executor>;

        ///////////////////////////////////////////////////////////////////////
        // If a parameters type exposes 'has_variable_chunk_size' aliased to
        // std::true_type it is assumed that the number of loop iterations to
        // combine is different for each of the generated chunks.
        template <typename Parameters>
        using extract_has_variable_chunk_size =
            execution::extract_has_variable_chunk_size<Parameters>;

        ///////////////////////////////////////////////////////////////////////
        template <typename Parameters_>
        struct processing_units_count_parameter_helper
        {
            template <typename Parameters, typename Executor>
            static std::size_t call(hpx::traits::detail::wrap_int,
                Parameters && params, Executor && exec)
            {
                return hpx::get_os_thread_count();
            }

            template <typename Parameters, typename Executor>
            static auto call(int, Parameters && params, Executor && exec)
            ->  decltype(params.processing_units_count(
                    std::forward<Executor>(exec)))
            {
                return params.processing_units_count(
                    std::forward<Executor>(exec));
            }

            template <typename Executor>
            static std::size_t call(Parameters_& params, Executor && exec)
            {
                return call(0, params, std::forward<Executor>(exec));
            }

            template <typename Parameters, typename Executor>
            static std::size_t call(Parameters params, Executor && exec)
            {
                return call(static_cast<Parameters_&>(params),
                    std::forward<Executor>(exec));
            }
        };

        template <typename Parameters, typename Executor>
        std::size_t call_processing_units_parameter_count(Parameters && params,
            Executor && exec)
        {
            return processing_units_count_parameter_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type
                >::call(std::forward<Parameters>(params),
                    std::forward<Executor>(exec));
        }

        template <typename T>
        using has_processing_units_count =
            execution::detail::has_count_processing_units<T>;

        ///////////////////////////////////////////////////////////////////////
        template <typename Parameters_>
        struct get_chunk_size_helper
        {
            template <typename Parameters, typename Executor, typename F>
            static std::size_t
            call(hpx::traits::detail::wrap_int, Parameters &&, Executor &&,
                F &&, std::size_t cores, std::size_t num_tasks)
            {
                return num_tasks;       // assume sequential execution
            }

            template <typename Parameters, typename Executor, typename F>
            static auto call(int, Parameters && params, Executor && exec,
                    F && f, std::size_t cores, std::size_t num_tasks)
            ->  decltype(
                    params.get_chunk_size(std::forward<Executor>(exec),
                        std::forward<F>(f), cores, num_tasks)
                )
            {
                return params.get_chunk_size(std::forward<Executor>(exec),
                    std::forward<F>(f), cores, num_tasks);
            }

            template <typename Executor, typename F>
            static std::size_t
            call(Parameters_& params, Executor && exec, F && f,
                std::size_t cores, std::size_t num_tasks)
            {
                return call(0, params, std::forward<Executor>(exec),
                    std::forward<F>(f), cores, num_tasks);
            }

            template <typename Parameters, typename Executor, typename F>
            static std::size_t
            call(Parameters params, Executor && exec, F && f,
                std::size_t cores, std::size_t num_tasks)
            {
                return call(static_cast<Parameters_&>(params),
                    std::forward<Executor>(exec), std::forward<F>(f),
                    cores, num_tasks);
            }
        };

        template <typename Parameters, typename Executor, typename F>
        std::size_t call_get_chunk_size(Parameters && params, Executor && exec,
            F && f, std::size_t cores, std::size_t num_tasks)
        {
            return get_chunk_size_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type
                >::call(std::forward<Parameters>(params),
                    std::forward<Executor>(exec), std::forward<F>(f),
                    cores, num_tasks);
        }

        template <typename T>
        using has_get_chunk_size = execution::detail::has_get_chunk_size<T>;

        ///////////////////////////////////////////////////////////////////////
        template <typename Parameters_>
        struct maximal_number_of_chunks_helper
        {
            template <typename Parameters, typename Executor>
            static std::size_t
            call(hpx::traits::detail::wrap_int, Parameters &&, Executor &&,
                std::size_t cores, std::size_t num_tasks)
            {
                return 4 * cores;       // assume 4 times the number of cores
            }

            template <typename Parameters, typename Executor>
            static auto call(int, Parameters && params, Executor && exec,
                    std::size_t cores, std::size_t num_tasks)
            ->  decltype(
                    params.maximal_number_of_chunks(
                        std::forward<Executor>(exec), cores, num_tasks)
                )
            {
                return params.maximal_number_of_chunks(
                    std::forward<Executor>(exec), cores, num_tasks);
            }

            template <typename Executor>
            static std::size_t
            call(Parameters_& params, Executor && exec, std::size_t cores,
                std::size_t num_tasks)
            {
                return call(0, params, std::forward<Executor>(exec), cores,
                    num_tasks);
            }

            template <typename Parameters, typename Executor>
            static std::size_t
            call(Parameters params, Executor && exec, std::size_t cores,
                std::size_t num_tasks)
            {
                return call(static_cast<Parameters_&>(params),
                    std::forward<Executor>(exec), cores, num_tasks);
            }
        };

        template <typename Parameters, typename Executor>
        std::size_t call_maximal_number_of_chunks(Parameters && params,
            Executor && exec, std::size_t cores, std::size_t num_tasks)
        {
            return maximal_number_of_chunks_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type
                >::call(std::forward<Parameters>(params),
                    std::forward<Executor>(exec), cores, num_tasks);
        }

        template <typename T>
        using has_maximal_number_of_chunks =
            execution::detail::has_maximal_number_of_chunks<T>;

        ///////////////////////////////////////////////////////////////////////
        template <typename Parameters_>
        struct reset_thread_distribution_helper
        {
            template <typename Parameters, typename Executor>
            static void call(hpx::traits::detail::wrap_int, Parameters &&,
                Executor &&)
            {
            }

            template <typename Parameters, typename Executor>
            static auto call(int, Parameters && params, Executor && exec)
            ->  decltype(
                    params.reset_thread_distribution(
                        std::forward<Executor>(exec))
                )
            {
                params.reset_thread_distribution(std::forward<Executor>(exec));
            }

            template <typename Executor>
            static void call(Parameters_& params, Executor && exec)
            {
                call(0, params, std::forward<Executor>(exec));
            }

            template <typename Parameters, typename Executor>
            static void call(Parameters params, Executor && exec)
            {
                call(static_cast<Parameters_&>(params),
                    std::forward<Executor>(exec));
            }
        };

        template <typename Parameters, typename Executor>
        void call_reset_thread_distribution(Parameters && params,
            Executor && exec)
        {
            reset_thread_distribution_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type
                >::call(std::forward<Parameters>(params),
                    std::forward<Executor>(exec));
        }

        template <typename T>
        using has_reset_thread_distribution =
            execution::detail::has_reset_thread_distribution<T>;

        ///////////////////////////////////////////////////////////////////////
        template <typename Parameters_>
        struct mark_begin_execution_helper
        {
            template <typename Parameters, typename Executor>
            static void call(hpx::traits::detail::wrap_int, Parameters &&,
                Executor &&)
            {
            }

            template <typename Parameters, typename Executor>
            static auto call(int, Parameters && params, Executor && exec)
            ->  decltype(params.mark_begin_execution(std::forward<Executor>(exec)))
            {
                params.mark_begin_execution(std::forward<Executor>(exec));
            }

            template <typename Executor>
            static void call(Parameters_& params, Executor && exec)
            {
                call(0, params, std::forward<Executor>(exec));
            }

            template <typename Parameters, typename Executor>
            static void call(Parameters params, Executor && exec)
            {
                call(static_cast<Parameters_&>(params),
                    std::forward<Executor>(exec));
            }
        };

        template <typename Parameters, typename Executor>
        void call_mark_begin_execution(Parameters && params, Executor && exec)
        {
            mark_begin_execution_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type
                >::call(std::forward<Parameters>(params),
                    std::forward<Executor>(exec));
        }

        template <typename T>
        using has_mark_begin_execution =
            execution::detail::has_mark_begin_execution<T>;

        ///////////////////////////////////////////////////////////////////////
        template <typename Parameters_>
        struct mark_end_execution_helper
        {
            template <typename Parameters, typename Executor>
            static void call(hpx::traits::detail::wrap_int, Parameters &&,
                Executor &&)
            {
            }

            template <typename Parameters, typename Executor>
            static auto call(int, Parameters && params, Executor && exec)
            ->  decltype(params.mark_end_execution(std::forward<Executor>(exec)))
            {
                params.mark_end_execution(std::forward<Executor>(exec));
            }

            template <typename Executor>
            static void call(Parameters_& params, Executor && exec)
            {
                call(0, params, std::forward<Executor>(exec));
            }

            template <typename Parameters, typename Executor>
            static void call(Parameters params, Executor && exec)
            {
                call(static_cast<Parameters_&>(params),
                    std::forward<Executor>(exec));
            }
        };

        template <typename Parameters, typename Executor>
        void call_mark_end_execution(Parameters && params, Executor && exec)
        {
            mark_end_execution_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type
                >::call(std::forward<Parameters>(params),
                    std::forward<Executor>(exec));
        }

        template <typename T>
        using has_mark_end_execution =
            execution::detail::has_mark_end_execution<T>;

        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The executor_parameter_traits type is used to manage parameters for
    /// an executor.
    template <typename Parameters, typename Enable>
    struct executor_parameter_traits
    {
        /// The type of the executor associated with this instance of
        /// \a executor_traits
        typedef Parameters executor_parameters_type;

        /// The compile-time information about whether the number of loop
        /// iterations to combine is different for each of the generated chunks.
        ///
        /// \note This calls extracts parameters_type::has_variable_chunk_size,
        ///       if available, otherwise it returns std::false_type.
        ///
        typedef typename detail::extract_has_variable_chunk_size<
                executor_parameters_type
            >::type has_variable_chunk_size;

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
        template <typename Parameters_, typename Executor, typename F>
        static std::size_t get_chunk_size(Parameters_ && params,
            Executor && exec, F && f, std::size_t cores, std::size_t num_tasks)
        {
            return detail::call_get_chunk_size(std::forward<Parameters_>(params),
                std::forward<Executor>(exec), std::forward<F>(f), cores, num_tasks);
        }

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
        template <typename Parameters_, typename Executor>
        static std::size_t maximal_number_of_chunks(
            Parameters_ && params, Executor && exec, std::size_t cores,
            std::size_t num_tasks)
        {
            return detail::call_maximal_number_of_chunks(
                std::forward<Parameters_>(params), std::forward<Executor>(exec),
                cores, num_tasks);
        }

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
        template <typename Parameters_, typename Executor>
        static void reset_thread_distribution(Parameters_ && params,
            Executor && exec)
        {
            detail::call_reset_thread_distribution(
                std::forward<Parameters_>(params), std::forward<Executor>(exec));
        }

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
        template <typename Parameters_, typename Executor>
        static std::size_t processing_units_count(Parameters_ && params,
            Executor && exec)
        {
            return detail::call_processing_units_parameter_count(
                std::forward<Parameters_>(params), std::forward<Executor>(exec));
        }

        /// Mark the begin of a parallel algorithm execution
        ///
        /// \param params [in] The executor parameters object to use as a
        ///              fallback if the executor does not expose
        ///
        /// \note This calls params.mark_begin_execution(exec) if it exists;
        ///       otherwise it does nothing.
        ///
        template <typename Parameters_, typename Executor>
        static void mark_begin_execution(Parameters_ && params, Executor && exec)
        {
            detail::call_mark_begin_execution(std::forward<Parameters_>(params),
                std::forward<Executor>(exec));
        }

        /// Mark the end of a parallel algorithm execution
        ///
        /// \param params [in] The executor parameters object to use as a
        ///              fallback if the executor does not expose
        ///
        /// \note This calls params.mark_end_execution(exec) if it exists;
        ///       otherwise it does nothing.
        ///
        template <typename Parameters_, typename Executor>
        static void mark_end_execution(Parameters_ && params, Executor && exec)
        {
            detail::call_mark_end_execution(std::forward<Parameters_>(params),
                std::forward<Executor>(exec));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // compatibility layer mapping new customization points onto
    // executor_information_traits

    // get_chunk_size()
    template <typename Parameters, typename Executor, typename F>
    HPX_FORCEINLINE
    typename std::enable_if<
        hpx::traits::is_executor_parameters<Parameters>::value &&
            hpx::traits::is_executor<Executor>::value,
        std::size_t
    >::type
    get_chunk_size(Parameters && params, Executor && exec, F && f,
        std::size_t cores, std::size_t num_tasks)
    {
        typedef typename std::decay<Parameters>::type parameter_type;
        return executor_parameter_traits<parameter_type>::get_chunk_size(
            std::forward<Parameters>(params), std::forward<Executor>(exec),
            std::forward<F>(f), cores, num_tasks);
    }

    // maximal_number_of_chunks()
    template <typename Parameters, typename Executor>
    HPX_FORCEINLINE
    typename std::enable_if<
        hpx::traits::is_executor_parameters<Parameters>::value &&
            hpx::traits::is_executor<Executor>::value,
        std::size_t
    >::type
    maximal_number_of_chunks(Parameters && params, Executor && exec,
        std::size_t cores, std::size_t num_tasks)
    {
        typedef typename std::decay<Parameters>::type parameter_type;
        return executor_parameter_traits<parameter_type>::
            maximal_number_of_chunks(std::forward<Parameters>(params),
                std::forward<Executor>(exec), cores, num_tasks);
    }

    // reset_thread_distribution()
    template <typename Parameters, typename Executor>
    HPX_FORCEINLINE
    typename std::enable_if<
        hpx::traits::is_executor_parameters<Parameters>::value &&
            hpx::traits::is_executor<Executor>::value
    >::type
    reset_thread_distribution(Parameters && params, Executor && exec)
    {
        typedef typename std::decay<Parameters>::type parameter_type;
        return executor_parameter_traits<parameter_type>::reset_thread_distribution(
            std::forward<Parameters>(params), std::forward<Executor>(exec));
    }

    // count_processing_units()
    template <typename Parameters, typename Executor>
    HPX_FORCEINLINE
    typename std::enable_if<
        hpx::traits::is_executor_parameters<Parameters>::value &&
            hpx::traits::is_executor<Executor>::value
    >::type
    count_processing_units(Parameters && params, Executor && exec)
    {
        typedef typename std::decay<Parameters>::type parameter_type;
        return executor_parameter_traits<parameter_type>::processing_units_count(
            std::forward<Parameters>(params), std::forward<Executor>(exec));
    }

    // mark_begin_execution()
    template <typename Parameters, typename Executor>
    HPX_FORCEINLINE
    typename std::enable_if<
        hpx::traits::is_executor_parameters<Parameters>::value &&
            hpx::traits::is_executor<Executor>::value
    >::type
    mark_begin_execution(Parameters && params, Executor && exec)
    {
        typedef typename std::decay<Parameters>::type parameter_type;
        return executor_parameter_traits<parameter_type>::mark_begin_execution(
            std::forward<Parameters>(params), std::forward<Executor>(exec));
    }

    // mark_end_execution()
    template <typename Parameters, typename Executor>
    HPX_FORCEINLINE
    typename std::enable_if<
        hpx::traits::is_executor_parameters<Parameters>::value &&
            hpx::traits::is_executor<Executor>::value
    >::type
    mark_end_execution(Parameters && params, Executor && exec)
    {
        typedef typename std::decay<Parameters>::type parameter_type;
        return executor_parameter_traits<parameter_type>::mark_end_execution(
            std::forward<Parameters>(params), std::forward<Executor>(exec));
    }
}}}

#endif
#endif
