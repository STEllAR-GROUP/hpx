//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_DATAPAR_EXECUTION_POLICY_SEP_07_2016_0601AM)
#define HPX_PARALLEL_DATAPAR_EXECUTION_POLICY_SEP_07_2016_0601AM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/datapar/execution_policy_fwd.hpp>
#include <hpx/parallel/executors/executor_parameter_traits.hpp>
#include <hpx/parallel/executors/executor_parameters.hpp>
#include <hpx/parallel/executors/sequenced_executor.hpp>
#include <hpx/parallel/executors/parallel_executor.hpp>
#include <hpx/parallel/executors/rebind_executor.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_execution_policy.hpp>
#include <hpx/traits/is_executor_v1.hpp>
#include <hpx/traits/is_launch_policy.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    /// Extension: The class dataseq_task_execution_policy is an execution
    /// policy type used as a unique type to disambiguate parallel algorithm
    /// overloading and indicate that a parallel algorithm's execution may not
    /// be parallelized (has to run sequentially).
    ///
    /// The algorithm returns a future representing the result of the
    /// corresponding algorithm when invoked with the
    /// sequential_execution_policy.
    struct dataseq_task_execution_policy
    {
        /// The type of the executor associated with this execution policy
        typedef parallel::sequential_executor executor_type;

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
            typedef dataseq_task_execution_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// \cond NOINTERNAL
        dataseq_task_execution_policy() {}
        /// \endcond

        /// Create a new dataseq_task_execution_policy from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequential_task_execution_policy
        ///
        dataseq_task_execution_policy operator()(
            task_execution_policy_tag tag) const
        {
            return *this;
        }

        /// Create a new dataseq_task_execution_policy from the given
        /// executor
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
        /// \returns The new dataseq_task_execution_policy
        ///
        template <typename Executor>
        typename rebind_executor<
            dataseq_task_execution_policy, Executor,
            executor_parameters_type
        >::type
        on(Executor && exec) const
        {
            static_assert(
                hpx::traits::is_executor<Executor>::value ||
                hpx::traits::is_threads_executor<Executor>::value,
                "hpx::traits::is_executor<Executor>::value || "
                "hpx::traits::is_threads_executor<Executor>::value");

            typedef typename rebind_executor<
                dataseq_task_execution_policy, Executor,
                executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor>(exec), parameters());
        }

        /// Create a new dataseq_task_execution_policy from the given
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
        /// \returns The new dataseq_task_execution_policy
        ///
        template <typename... Parameters, typename ParametersType =
            typename executor_parameters_join<Parameters...>::type>
        typename rebind_executor<
            dataseq_task_execution_policy, executor_type, ParametersType
        >::type
        with(Parameters &&... params) const
        {
            typedef typename rebind_executor<
                dataseq_task_execution_policy, executor_type, ParametersType
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

    /// Extension: The class dataseq_task_execution_policy_shim is an
    /// execution policy type used as a unique type to disambiguate parallel
    /// algorithm overloading based on combining a underlying
    /// \a sequential_task_execution_policy and an executor and indicate that
    /// a parallel algorithm's execution may not be parallelized  (has to run
    /// sequentially).
    ///
    /// The algorithm returns a future representing the result of the
    /// corresponding algorithm when invoked with the
    /// sequential_execution_policy.
    template <typename Executor, typename Parameters>
    struct dataseq_task_execution_policy_shim
      : dataseq_task_execution_policy
    {
        /// The type of the executor associated with this execution policy
        typedef Executor executor_type;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        typedef Parameters executor_parameters_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef typename executor_traits<executor_type>::execution_category
            execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef dataseq_task_execution_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// Create a new dataseq_task_execution_policy_shim from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequential_task_execution_policy
        ///
        dataseq_task_execution_policy_shim const& operator()(
            task_execution_policy_tag tag) const
        {
            return *this;
        }

        /// Create a new dataseq_task_execution_policy_shim from the given
        /// executor
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
        /// \returns The new dataseq_task_execution_policy_shim
        ///
        template <typename Executor_>
        typename rebind_executor<
            dataseq_task_execution_policy_shim, Executor_,
            executor_parameters_type
        >::type
        on(Executor_ && exec) const
        {
            static_assert(
                hpx::traits::is_executor<Executor_>::value ||
                hpx::traits::is_threads_executor<Executor_>::value,
                "hpx::traits::is_executor<Executor_>::value || "
                "hpx::traits::is_threads_executor<Executor_>::value");

            typedef typename rebind_executor<
                dataseq_task_execution_policy_shim, Executor_,
                executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor_>(exec), params_);
        }

        /// Create a new dataseq_task_execution_policy_shim from the given
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
        /// \returns The new sequential_task_execution_policy_shim
        ///
        template <typename... Parameters_, typename ParametersType =
            typename executor_parameters_join<Parameters_...>::type>
        typename rebind_executor<
            dataseq_task_execution_policy_shim, executor_type, ParametersType
        >::type
        with(Parameters_ &&... params) const
        {
            typedef typename rebind_executor<
                dataseq_task_execution_policy_shim, executor_type,
                ParametersType
            >::type rebound_type;
            return rebound_type(exec_,
                join_executor_parameters(std::forward<Parameters_>(params)...));
        }

        /// Return the associated executor object.
        Executor& executor() { return exec_; }
        /// Return the associated executor object.
        Executor const& executor() const { return exec_; }

        /// Return the associated executor parameters object.
        Parameters& parameters() { return params_; }
        /// Return the associated executor parameters object.
        Parameters const& parameters() const { return params_; }

        /// \cond NOINTERNAL
        dataseq_task_execution_policy_shim() {}

        template <typename Executor_, typename Parameters_>
        dataseq_task_execution_policy_shim(
                Executor_ && exec, Parameters_ && params)
          : exec_(std::forward<Executor_>(exec)),
            params_(std::forward<Parameters_>(params))
        {}

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & exec_ & params_;
        }

    private:
        Executor exec_;
        Parameters params_;
        /// \endcond
    };

    ///////////////////////////////////////////////////////////////////////////
    /// The class dataseq_execution_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// require that a parallel algorithm's execution may not be parallelized.
    struct dataseq_execution_policy
    {
        /// The type of the executor associated with this execution policy
        typedef parallel::sequential_executor executor_type;

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
            typedef dataseq_execution_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// \cond NOINTERNAL
        dataseq_execution_policy() {}
        /// \endcond

        /// Create a new dataseq_task_execution_policy.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new dataseq_task_execution_policy
        ///
        dataseq_task_execution_policy operator()(
            task_execution_policy_tag tag) const
        {
            return dataseq_task_execution_policy();
        }

        /// Create a new dataseq_execution_policy from the given
        /// executor
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
        /// \returns The new dataseq_execution_policy
        ///
        template <typename Executor>
        typename rebind_executor<
            dataseq_execution_policy, Executor, executor_parameters_type
        >::type
        on(Executor && exec) const
        {
            static_assert(
                hpx::traits::is_executor<Executor>::value ||
                hpx::traits::is_threads_executor<Executor>::value,
                "hpx::traits::is_executor<Executor>::value || "
                "hpx::traits::is_threads_executor<Executor>::value");

            typedef typename rebind_executor<
                dataseq_execution_policy, Executor, executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor>(exec), parameters());
        }

        /// Create a new dataseq_execution_policy from the given
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
        /// \returns The new dataseq_execution_policy
        ///
        template <typename... Parameters, typename ParametersType =
            typename executor_parameters_join<Parameters...>::type>
        typename rebind_executor<
            dataseq_execution_policy, executor_type, ParametersType
        >::type
        with(Parameters &&... params) const
        {
            typedef typename rebind_executor<
                dataseq_execution_policy, executor_type, ParametersType
            >::type rebound_type;
            return rebound_type(executor(),
                join_executor_parameters(std::forward<Parameters>(params)...));
        }

    public:
        /// Return the associated executor object.
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

    /// Default sequential execution policy object.
    static dataseq_execution_policy const dataseq_execution;

    /// The class dataseq_execution_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// require that a parallel algorithm's execution may not be parallelized.
    template <typename Executor, typename Parameters>
    struct dataseq_execution_policy_shim : dataseq_execution_policy
    {
        /// The type of the executor associated with this execution policy
        typedef Executor executor_type;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        typedef Parameters executor_parameters_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef typename executor_traits<executor_type>::execution_category
            execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef dataseq_execution_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// Create a new dataseq_task_execution_policy.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new dataseq_task_execution_policy_shim
        ///
        dataseq_task_execution_policy_shim<Executor, Parameters> operator()(
            task_execution_policy_tag tag) const
        {
            return dataseq_task_execution_policy_shim<
                Executor, Parameters>(exec_, params_);
        }

        /// Create a new dataseq_execution_policy from the given
        /// executor
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
        /// \returns The new dataseq_execution_policy
        ///
        template <typename Executor_>
        typename rebind_executor<
            dataseq_execution_policy_shim, Executor_,
            executor_parameters_type
        >::type
        on(Executor_ && exec) const
        {
            static_assert(
                hpx::traits::is_executor<Executor_>::value ||
                hpx::traits::is_threads_executor<Executor_>::value,
                "hpx::traits::is_executor<Executor_>::value || "
                "hpx::traits::is_threads_executor<Executor_>::value");

            typedef typename rebind_executor<
                dataseq_execution_policy_shim, Executor_,
                executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor_>(exec), params_);
        }

        /// Create a new dataseq_execution_policy_shim from the given
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
        /// \returns The new dataseq_execution_policy_shim
        ///
        template <typename... Parameters_, typename ParametersType =
            typename executor_parameters_join<Parameters_...>::type>
        typename rebind_executor<
            dataseq_execution_policy_shim, executor_type, ParametersType
        >::type
        with(Parameters_ &&... params) const
        {
            typedef typename rebind_executor<
                dataseq_execution_policy_shim, executor_type, ParametersType
            >::type rebound_type;
            return rebound_type(exec_,
                join_executor_parameters(std::forward<Parameters_>(params)...));
        }

        /// Return the associated executor object.
        Executor& executor() { return exec_; }
        /// Return the associated executor object.
        Executor const& executor() const { return exec_; }

        /// Return the associated executor parameters object.
        Parameters& parameters() { return params_; }
        /// Return the associated executor parameters object.
        Parameters const& parameters() const { return params_; }

        /// \cond NOINTERNAL
        dataseq_execution_policy_shim() {}

        template <typename Executor_, typename Parameters_>
        dataseq_execution_policy_shim(
                Executor_ && exec, Parameters_ && params)
          : exec_(std::forward<Executor_>(exec)),
            params_(std::forward<Parameters_>(params))
        {}

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & exec_ & params_;
        }

    private:
        Executor exec_;
        Parameters params_;
        /// \endcond
    };

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
        /// \returns The new datapar_execution_policy_shim
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

    /// The class datapar_execution_policy_shim is an execution policy type
    /// used as a unique type to disambiguate parallel algorithm overloading
    /// and indicate that a parallel algorithm's execution may be parallelized.
    template <typename Executor, typename Parameters>
    struct datapar_execution_policy_shim : datapar_execution_policy
    {
        /// The type of the executor associated with this execution policy
        typedef Executor executor_type;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        typedef Parameters executor_parameters_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef typename executor_traits<executor_type>::execution_category
            execution_category;

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

        /// Create a new datapar_task_execution_policy_shim
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new datapar_task_execution_policy_shim
        ///
        datapar_task_execution_policy_shim<Executor, Parameters>
        operator()(task_execution_policy_tag tag) const
        {
            return datapar_task_execution_policy_shim<Executor, Parameters>(
                exec_, params_);
        }

        /// Create a new datapar_task_execution_policy_shim from the given
        /// executor
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
        /// \returns The new parallel_execution_policy
        ///
        template <typename Executor_>
        typename rebind_executor<
            datapar_execution_policy_shim, Executor_,
            executor_parameters_type
        >::type
        on(Executor_ && exec) const
        {
            static_assert(
                hpx::traits::is_executor<Executor_>::value ||
                hpx::traits::is_threads_executor<Executor_>::value,
                "hpx::traits::is_executor<Executor_>::value || "
                "hpx::traits::is_threads_executor<Executor_>::value");

            typedef typename rebind_executor<
                datapar_execution_policy_shim, Executor_,
                executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor_>(exec), params_);
        }

        /// Create a new datapar_execution_policy_shim from the given
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
        /// \returns The new datapar_execution_policy_shim
        ///
        template <typename... Parameters_, typename ParametersType =
            typename executor_parameters_join<Parameters_...>::type>
        typename rebind_executor<
            datapar_execution_policy_shim, executor_type, ParametersType
        >::type
        with(Parameters_ &&... params) const
        {
            typedef typename rebind_executor<
                datapar_execution_policy_shim, executor_type, ParametersType
            >::type rebound_type;
            return rebound_type(exec_,
                join_executor_parameters(std::forward<Parameters_>(params)...));
        }

        /// Return the associated executor object.
        Executor& executor() { return exec_; }
        /// Return the associated executor object.
        Executor const& executor() const { return exec_; }

        /// Return the associated executor parameters object.
        Parameters& parameters() { return params_; }
        /// Return the associated executor parameters object.
        Parameters const& parameters() const { return params_; }

        /// \cond NOINTERNAL
        datapar_execution_policy_shim() {}

        template <typename Executor_, typename Parameters_>
        datapar_execution_policy_shim(
                Executor_ && exec, Parameters_ && params)
          : exec_(std::forward<Executor_>(exec)),
            params_(std::forward<Parameters_>(params))
        {}

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & exec_ & params_;
        }

    private:
        Executor exec_;
        Parameters params_;
        /// \endcond
    };

    /// The class datapar_task_execution_policy_shim is an
    /// execution policy type used as a unique type to disambiguate parallel
    /// algorithm overloading based on combining a underlying
    /// \a datapar_task_execution_policy and an executor and indicate that
    /// a parallel algorithm's execution may be parallelized and vectorized.
    template <typename Executor, typename Parameters>
    struct datapar_task_execution_policy_shim : datapar_task_execution_policy
    {
        /// The type of the executor associated with this execution policy
        typedef Executor executor_type;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        typedef Parameters executor_parameters_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef typename executor_traits<executor_type>::execution_category
            execution_category;

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

        /// Create a new datapar_task_execution_policy_shim from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequential_task_execution_policy
        ///
        datapar_task_execution_policy_shim operator()(
            task_execution_policy_tag tag) const
        {
            return *this;
        }

        /// Create a new parallel_task_execution_policy from the given
        /// executor
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
        /// \returns The new parallel_task_execution_policy
        ///
        template <typename Executor_>
        typename rebind_executor<
            datapar_task_execution_policy_shim, Executor_,
            executor_parameters_type
        >::type
        on(Executor_ && exec) const
        {
            static_assert(
                hpx::traits::is_executor<Executor_>::value ||
                hpx::traits::is_threads_executor<Executor_>::value,
                "hpx::traits::is_executor<Executor_>::value || "
                "hpx::traits::is_threads_executor<Executor_>::value");

            typedef typename rebind_executor<
                datapar_task_execution_policy_shim, Executor_,
                executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor_>(exec), params_);
        }

        /// Create a new parallel_execution_policy_shim from the given
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
        template <typename... Parameters_, typename ParametersType =
            typename executor_parameters_join<Parameters_...>::type>
        typename rebind_executor<
            datapar_task_execution_policy_shim, executor_type, ParametersType
        >::type
        with(Parameters_ &&... params) const
        {
            typedef typename rebind_executor<
                datapar_task_execution_policy_shim, executor_type, ParametersType
            >::type rebound_type;
            return rebound_type(exec_,
                join_executor_parameters(std::forward<Parameters_>(params)...));
        }

        /// Return the associated executor object.
        Executor& executor() { return exec_; }
        /// Return the associated executor object.
        Executor const& executor() const { return exec_; }

        /// Return the associated executor parameters object.
        Parameters& parameters() { return params_; }
        /// Return the associated executor parameters object.
        Parameters const& parameters() const { return params_; }

        /// \cond NOINTERNAL
        datapar_task_execution_policy_shim() {}

        template <typename Executor_, typename Parameters_>
        datapar_task_execution_policy_shim(
                Executor_ && exec, Parameters_ && params)
          : exec_(std::forward<Executor_>(exec)),
            params_(std::forward<Parameters_>(params))
        {}

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & exec_ & params_;
        }

    private:
        Executor exec_;
        Parameters params_;
        /// \endcond
    };

    ///////////////////////////////////////////////////////////////////////////
    // extensions
    namespace detail
    {
        /// \cond NOINTERNAL
        template <>
        struct is_execution_policy<dataseq_execution_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_execution_policy<
                dataseq_execution_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <>
        struct is_execution_policy<dataseq_task_execution_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_execution_policy<
                dataseq_task_execution_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <>
        struct is_execution_policy<datapar_execution_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_execution_policy<
                datapar_execution_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <>
        struct is_execution_policy<datapar_task_execution_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_execution_policy<
                datapar_task_execution_policy_shim<Executor, Parameters> >
          : std::true_type
        {};
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <>
        struct is_sequential_execution_policy<dataseq_execution_policy>
          : std::true_type
        {};

        template <>
        struct is_sequential_execution_policy<dataseq_task_execution_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_sequential_execution_policy<
                dataseq_execution_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_sequential_execution_policy<
                dataseq_task_execution_policy_shim<Executor, Parameters> >
          : std::true_type
        {};
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <>
        struct is_async_execution_policy<dataseq_task_execution_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_async_execution_policy<
                dataseq_task_execution_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <>
        struct is_async_execution_policy<datapar_task_execution_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_async_execution_policy<
                datapar_task_execution_policy_shim<Executor, Parameters> >
          : std::true_type
        {};
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <>
        struct is_parallel_execution_policy<datapar_execution_policy>
          : std::true_type
        {};

        template <>
        struct is_parallel_execution_policy<datapar_task_execution_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_parallel_execution_policy<
                datapar_execution_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_parallel_execution_policy<
                datapar_task_execution_policy_shim<Executor, Parameters> >
          : std::true_type
        {};
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <>
        struct is_vectorpack_execution_policy<dataseq_execution_policy>
          : std::true_type
        {};

        template <>
        struct is_vectorpack_execution_policy<dataseq_task_execution_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_vectorpack_execution_policy<
                dataseq_execution_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_vectorpack_execution_policy<
                dataseq_task_execution_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <>
        struct is_vectorpack_execution_policy<datapar_execution_policy>
          : std::true_type
        {};

        template <>
        struct is_vectorpack_execution_policy<datapar_task_execution_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_vectorpack_execution_policy<
                datapar_execution_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_vectorpack_execution_policy<
                datapar_task_execution_policy_shim<Executor, Parameters> >
          : std::true_type
        {};
        /// \endcond
    }
}}}

#endif
#endif
