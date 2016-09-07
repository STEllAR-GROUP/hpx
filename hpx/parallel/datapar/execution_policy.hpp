//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_EXECUTION_POLICY_SEP_07_2016_0601AM)
#define HPX_PARALLEL_DATAPAR_EXECUTION_POLICY_SEP_07_2016_0601AM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_VC_DATAPAR)
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/datapar/execution_policy_fwd.hpp>
#include <hpx/parallel/executors/parallel_executor.hpp>
#include <hpx/parallel/executors/rebind_executor.hpp>
#include <hpx/parallel/executors/executor_parameter_traits.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_execution_policy.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/traits/is_launch_policy.hpp>

#include <utility>

#include <Vc/Vc>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    /// Extension: The class datapar_task_execution_policy is an execution
    /// policy type used as a unique type to disambiguate parallel algorithm
    /// overloading and indicate that a parallel algorithm's execution may be
    /// parallelized.
    ///
    /// The algorithm returns a future representing the result of the
    /// corresponding algorithm when invoked with the parallel_execution_policy.
    struct datapar_task_execution_policy
    {
        /// The type of the executor associated with this execution policy
        typedef parallel::parallel_executor executor_type;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        typedef v3::detail::extract_executor_parameters<
                executor_type
            >::type executor_parameters_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef parallel::vector_execution_tag execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef datapar_task_execution_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// \cond NOINTERNAL
        datapar_task_execution_policy() {}
        /// \endcond

        /// Create a new datapar_task_execution_policy from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new datapar_task_execution_policy
        ///
        datapar_task_execution_policy operator()(
            task_execution_policy_tag tag) const
        {
            return *this;
        }

        /// Create a new datapar_task_execution_policy from given executor
        ///
        /// \tparam Executor    The type of the executor to associate with this
        ///                     execution policy.
        ///
        /// \param exec         [in] The executor to use for the
        ///                     execution of the parallel algorithm the
        ///                     returned execution policy is used with.
        ///
        /// \note Requires: is_executor<Executor>::value is true
        ///
        /// \returns The new datapar_task_execution_policy
        ///
        template <typename Executor>
        typename rebind_executor<
            datapar_task_execution_policy, Executor, executor_parameters_type
        >::type
        on(Executor && exec) const
        {
            static_assert(
                hpx::traits::is_executor<Executor>::value ||
                hpx::traits::is_threads_executor<Executor>::value,
                "hpx::traits::is_executor<Executor>::value || "
                "hpx::traits::is_threads_executor<Executor>::value");

            typedef typename rebind_executor<
                datapar_task_execution_policy, Executor,
                executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor>(exec), parameters());
        }

        /// Create a new datapar_task_execution_policy_shim from the given
        /// execution parameters
        ///
        /// \tparam Parameters  The type of the executor parameters to
        ///                     associate with this execution policy.
        ///
        /// \param params       [in] The executor parameters to use for the
        ///                     execution of the parallel algorithm the
        ///                     returned execution policy is used with.
        ///
        /// \note Requires: all parameters are executor_parameters,
        ///                 different parameter types can't be duplicated
        ///
        /// \returns The new parallel_execution_policy_shim
        ///
        template <typename... Parameters, typename ParametersType =
            typename executor_parameters_join<Parameters...>::type>
        typename rebind_executor<
            datapar_task_execution_policy, executor_type, ParametersType
        >::type
        with(Parameters &&... params) const
        {
            typedef typename rebind_executor<
                datapar_task_execution_policy, executor_type, ParametersType
            >::type rebound_type;
            return rebound_type(executor(),
                join_executor_parameters(std::forward<Parameters>(params)...));
        }

    public:
        /// Return the associated executor object.
        executor_type& executor() { return exec_; }
        /// Return the associated executor object.
        executor_type const& executor() const { return exec_; }

        /// Return the associated executor parameters object.
        executor_parameters_type& parameters() { return params_; }
        /// Return the associated executor parameters object.
        executor_parameters_type const& parameters() const { return params_; }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
        }

    private:
        executor_type exec_;
        executor_parameters_type params_;
    };

    ///////////////////////////////////////////////////////////////////////////
    /// The class datapar_execution_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// indicate that a parallel algorithm's execution may be parallelized.
    struct datapar_execution_policy
    {
        /// The type of the executor associated with this execution policy
        typedef parallel::parallel_executor executor_type;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        typedef v3::detail::extract_executor_parameters<
                executor_type
            >::type executor_parameters_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef parallel::vector_execution_tag execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef datapar_execution_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// \cond NOINTERNAL
        datapar_execution_policy() {}
        /// \endcond

        /// Create a new datapar_execution_policy referencing a chunk size.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new datapar_execution_policy
        ///
        datapar_task_execution_policy operator()(
            task_execution_policy_tag tag) const
        {
            return datapar_task_execution_policy();
        }

        /// Create a new datapar_execution_policy referencing an executor and
        /// a chunk size.
        ///
        /// \param exec         [in] The executor to use for the execution of
        ///                     the parallel algorithm the returned execution
        ///                     policy is used with
        ///
        /// \returns The new datapar_execution_policy
        ///
        template <typename Executor>
        typename rebind_executor<
            datapar_execution_policy, Executor, executor_parameters_type
        >::type
        on(Executor && exec) const
        {
            static_assert(
                hpx::traits::is_executor<Executor>::value ||
                hpx::traits::is_threads_executor<Executor>::value,
                "hpx::traits::is_executor<Executor>::value || "
                "hpx::traits::is_threads_executor<Executor>::value");

            typedef typename rebind_executor<
                datapar_execution_policy, Executor, executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor>(exec), parameters());
        }

        /// Create a new datapar_execution_policy from the given
        /// execution parameters
        ///
        /// \tparam Parameters  The type of the executor parameters to
        ///                     associate with this execution policy.
        ///
        /// \param params       [in] The executor parameters to use for the
        ///                     execution of the parallel algorithm the
        ///                     returned execution policy is used with.
        ///
        /// \note Requires: is_executor_parameters<Parameters>::value is true
        ///
        /// \returns The new datapar_execution_policy
        ///
        template <typename... Parameters, typename ParametersType =
            typename executor_parameters_join<Parameters...>::type>
        typename rebind_executor<
            datapar_execution_policy, executor_type, ParametersType
        >::type
        with(Parameters &&... params) const
        {
            typedef typename rebind_executor<
                datapar_execution_policy, executor_type, ParametersType
            >::type rebound_type;
            return rebound_type(executor(),
                join_executor_parameters(std::forward<Parameters>(params)...));
        }

    public:
        /// Return the associated executor object.
        executor_type& executor() { return exec_; }
        /// Return the associated executor object.
        executor_type const& executor() const { return exec_; }

        /// Return the associated executor parameters object.
        executor_parameters_type& parameters() { return params_; }
        /// Return the associated executor parameters object.
        executor_parameters_type const& parameters() const { return params_; }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
        }

    private:
        executor_type exec_;
        executor_parameters_type params_;
    };

    /// Default data-parallel execution policy object.
    static datapar_execution_policy const datapar_execution;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor, typename Parameters>
    struct datapar_execution_policy_shim;

    template <typename Executor, typename Parameters>
    struct datapar_task_execution_policy_shim;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // extension
        template <>
        struct is_execution_policy<datapar_execution_policy>
          : std::true_type
        {};

        template <>
        struct is_execution_policy<datapar_task_execution_policy>
          : std::true_type
        {};

        template <>
        struct is_parallel_execution_policy<datapar_execution_policy>
          : std::true_type
        {};

        template <>
        struct is_parallel_execution_policy<datapar_task_execution_policy>
          : std::true_type
        {};

        template <>
        struct is_async_execution_policy<datapar_task_execution_policy>
          : std::true_type
        {};
    }
}}}

#endif
#endif
