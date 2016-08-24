//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/executor_parameter_traits.hpp

#if !defined(HPX_PARALLEL_EXECUTOR_PARAMETER_TRAITS_JUL_30_2015_0914PM)
#define HPX_PARALLEL_EXECUTOR_PARAMETER_TRAITS_JUL_30_2015_0914PM

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/traits/has_member_xxx.hpp>
#include <hpx/traits/is_executor_parameters.hpp>
#include <hpx/traits/detail/wrap_int.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/decay.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    // Placeholder type to use predefined executor parameters
    struct sequential_executor_parameters : executor_parameters_tag {};

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    namespace detail
    {
        // If an executor exposes 'executor_parameter_type' this type is
        // assumed to represent the default parameters for the given executor
        // type.
        template <typename Executor, typename Enable = void>
        struct extract_executor_parameters
        {
            // by default, assume sequential execution
            typedef sequential_executor_parameters type;
        };

        template <typename Executor>
        struct extract_executor_parameters<Executor,
            typename hpx::util::always_void<
                typename Executor::executor_parameters_type
            >::type>
        {
            typedef typename Executor::executor_parameters_type type;
        };

        ///////////////////////////////////////////////////////////////////////
        // If a parameters type exposes 'has_variable_chunk_size' aliased to
        // std::true_type it is assumed that the number of loop iterations to
        // combine is different for each of the generated chunks.
        template <typename Parameters, typename Enable = void>
        struct extract_has_variable_chunk_size
        {
            // by default, assume equally sized chunks
            typedef std::false_type type;
        };

        template <typename Parameters>
        struct extract_has_variable_chunk_size<Parameters,
            typename hpx::util::always_void<
                typename Parameters::has_variable_chunk_size
            >::type>
        {
            typedef typename Parameters::has_variable_chunk_size type;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Parameters_>
        struct processing_units_count_parameter_helper
        {
            template <typename Parameters>
            static std::size_t call(hpx::traits::detail::wrap_int,
                Parameters && params)
            {
                return hpx::get_os_thread_count();
            }

            template <typename Parameters>
            static auto call(int, Parameters && params)
            ->  decltype(params.processing_units_count())
            {
                return params.processing_units_count();
            }

            static std::size_t call(Parameters_& params)
            {
                return call(0, params);
            }

            template <typename Parameters>
            static std::size_t call(Parameters params)
            {
                return call(static_cast<Parameters_&>(params));
            }
        };

        template <typename Parameters>
        std::size_t call_processing_units_parameter_count(Parameters && params)
        {
            return processing_units_count_parameter_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type
                >::call(std::forward<Parameters>(params));
        }

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(processing_units_count);

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

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(get_chunk_size);

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

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(maximal_number_of_chunks);

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

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(reset_thread_distribution);

        ///////////////////////////////////////////////////////////////////////
        template <typename Parameters_>
        struct mark_begin_execution_helper
        {
            template <typename Parameters>
            static void call(hpx::traits::detail::wrap_int, Parameters &&)
            {
            }

            template <typename Parameters>
            static auto call(int, Parameters && params)
            ->  decltype(params.mark_begin_execution())
            {
                params.mark_begin_execution();
            }

            static void call(Parameters_& params)
            {
                call(0, params);
            }

            template <typename Parameters>
            static void call(Parameters params)
            {
                call(static_cast<Parameters_&>(params));
            }
        };

        template <typename Parameters>
        void call_mark_begin_execution(Parameters && params)
        {
            mark_begin_execution_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type
                >::call(std::forward<Parameters>(params));
        }

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(mark_begin_execution);

        ///////////////////////////////////////////////////////////////////////
        template <typename Parameters_>
        struct mark_end_execution_helper
        {
            template <typename Parameters>
            static void call(hpx::traits::detail::wrap_int, Parameters &&)
            {
            }

            template <typename Parameters>
            static auto call(int, Parameters && params)
            ->  decltype(params.mark_end_execution())
            {
                params.mark_end_execution();
            }

            static void call(Parameters_& params)
            {
                call(0, params);
            }

            template <typename Parameters>
            static void call(Parameters params)
            {
                call(static_cast<Parameters_&>(params));
            }
        };

        template <typename Parameters>
        void call_mark_end_execution(Parameters && params)
        {
            mark_end_execution_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type
                >::call(std::forward<Parameters>(params));
        }

        HPX_HAS_MEMBER_XXX_TRAIT_DEF(mark_end_execution);
    }
    /// \endcond

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
        /// \param exec     [in] The executor object which will be used used
        ///                 for scheduling of the the loop iterations.
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
        /// \param exec     [in] The executor object which will be used used
        ///                 for scheduling of the the loop iterations.
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
        template <typename Parameters_>
        static std::size_t processing_units_count(Parameters_ && params)
        {
            return detail::call_processing_units_parameter_count(
                std::forward<Parameters_>(params));
        }

        /// Mark the begin of a parallel algorithm execution
        ///
        /// \param params [in] The executor parameters object to use as a
        ///              fallback if the executor does not expose
        ///
        /// \note This calls params.mark_begin_execution(exec) if it exists;
        ///       otherwise it does nothing.
        ///
        template <typename Parameters_>
        static void mark_begin_execution(Parameters_ && params)
        {
            detail::call_mark_begin_execution(std::forward<Parameters_>(params));
        }

        /// Mark the end of a parallel algorithm execution
        ///
        /// \param params [in] The executor parameters object to use as a
        ///              fallback if the executor does not expose
        ///
        /// \note This calls params.mark_end_execution(exec) if it exists;
        ///       otherwise it does nothing.
        ///
        template <typename Parameters_>
        static void mark_end_execution(Parameters_ && params)
        {
            detail::call_mark_end_execution(std::forward<Parameters_>(params));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // defined in hpx/traits/is_executor_parameters.hpp
    ///
    /// 1. The type is_executor_parameters can be used to detect executor
    ///    parameters types for the purpose of excluding function signatures
    ///    from otherwise ambiguous overload resolution participation.
    /// 2. If T is the type of a standard or implementation-defined executor,
    ///    is_executor_parameters<T> shall be publicly derived from
    ///    integral_constant<bool, true>, otherwise from
    ///    integral_constant<bool, false>.
    /// 3. The behavior of a program that adds specializations for
    ///    is_executor_parameters is undefined.
    ///
    template <typename T>
    struct is_executor_parameters;
}}}

#endif
