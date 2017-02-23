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
#include <hpx/parallel/executors/sequential_executor.hpp>
#include <hpx/parallel/executors/parallel_executor.hpp>
#include <hpx/parallel/executors/rebind_executor.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_execution_policy.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/traits/is_launch_policy.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace execution { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    /// Extension: The class dataseq_task_policy is an execution
    /// policy type used as a unique type to disambiguate parallel algorithm
    /// overloading and indicate that a parallel algorithm's execution may not
    /// be parallelized (has to run sequentially).
    ///
    /// The algorithm returns a future representing the result of the
    /// corresponding algorithm when invoked with the
    /// sequenced_policy.
    struct dataseq_task_policy
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
            typedef dataseq_task_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// \cond NOINTERNAL
        HPX_CONSTEXPR dataseq_task_policy() {}
        /// \endcond

        /// Create a new dataseq_task_policy from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequenced_task_policy
        ///
        HPX_CONSTEXPR dataseq_task_policy operator()(task_policy_tag tag) const
        {
            return *this;
        }

        /// Create a new dataseq_task_policy from the given
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
        /// \returns The new dataseq_task_policy
        ///
        template <typename Executor>
        typename rebind_executor<
            dataseq_task_policy, Executor,
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
                dataseq_task_policy, Executor,
                executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor>(exec), parameters());
        }

        /// Create a new dataseq_task_policy from the given
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
        /// \returns The new dataseq_task_policy
        ///
        template <typename... Parameters, typename ParametersType =
            typename executor_parameters_join<Parameters...>::type>
        typename rebind_executor<
            dataseq_task_policy, executor_type, ParametersType
        >::type
        with(Parameters &&... params) const
        {
            typedef typename rebind_executor<
                dataseq_task_policy, executor_type, ParametersType
            >::type rebound_type;
            return rebound_type(executor(),
                join_executor_parameters(std::forward<Parameters>(params)...));
        }

    public:
        /// Return the associated executor object.
        executor_type& executor() { return exec_; }
        /// Return the associated executor object.
        HPX_CONSTEXPR executor_type const& executor() const { return exec_; }

        /// Return the associated executor parameters object.
        executor_parameters_type& parameters() { return params_; }
        /// Return the associated executor parameters object.
        HPX_CONSTEXPR executor_parameters_type const& parameters() const
            { return params_; }

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

    /// Extension: The class dataseq_task_policy_shim is an
    /// execution policy type used as a unique type to disambiguate parallel
    /// algorithm overloading based on combining a underlying
    /// \a sequenced_task_policy and an executor and indicate that
    /// a parallel algorithm's execution may not be parallelized  (has to run
    /// sequentially).
    ///
    /// The algorithm returns a future representing the result of the
    /// corresponding algorithm when invoked with the
    /// sequenced_policy.
    template <typename Executor, typename Parameters>
    struct dataseq_task_policy_shim
      : dataseq_task_policy
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
            typedef dataseq_task_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// Create a new dataseq_task_policy_shim from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequenced_task_policy
        ///
        HPX_CONSTEXPR dataseq_task_policy_shim const& operator()(
            task_policy_tag tag) const
        {
            return *this;
        }

        /// Create a new dataseq_task_policy_shim from the given
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
        /// \returns The new dataseq_task_policy_shim
        ///
        template <typename Executor_>
        typename rebind_executor<
            dataseq_task_policy_shim, Executor_,
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
                dataseq_task_policy_shim, Executor_,
                executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor_>(exec), params_);
        }

        /// Create a new dataseq_task_policy_shim from the given
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
        /// \returns The new sequenced_task_policy_shim
        ///
        template <typename... Parameters_, typename ParametersType =
            typename executor_parameters_join<Parameters_...>::type>
        typename rebind_executor<
            dataseq_task_policy_shim, executor_type, ParametersType
        >::type
        with(Parameters_ &&... params) const
        {
            typedef typename rebind_executor<
                dataseq_task_policy_shim, executor_type,
                ParametersType
            >::type rebound_type;
            return rebound_type(exec_,
                join_executor_parameters(std::forward<Parameters_>(params)...));
        }

        /// Return the associated executor object.
        Executor& executor() { return exec_; }
        /// Return the associated executor object.
        HPX_CONSTEXPR Executor const& executor() const { return exec_; }

        /// Return the associated executor parameters object.
        Parameters& parameters() { return params_; }
        /// Return the associated executor parameters object.
        HPX_CONSTEXPR Parameters const& parameters() const { return params_; }

        /// \cond NOINTERNAL
        HPX_CONSTEXPR dataseq_task_policy_shim() {}

        template <typename Executor_, typename Parameters_>
        HPX_CONSTEXPR dataseq_task_policy_shim(
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
    /// The class dataseq_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// require that a parallel algorithm's execution may not be parallelized.
    struct dataseq_policy
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
            typedef dataseq_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// \cond NOINTERNAL
        HPX_CONSTEXPR dataseq_policy() : exec_{}, params_{} {}
        /// \endcond

        /// Create a new dataseq_task_policy.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new dataseq_task_policy
        ///
        HPX_CONSTEXPR dataseq_task_policy operator()(task_policy_tag tag) const
        {
            return dataseq_task_policy();
        }

        /// Create a new dataseq_policy from the given
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
        /// \returns The new dataseq_policy
        ///
        template <typename Executor>
        typename rebind_executor<
            dataseq_policy, Executor, executor_parameters_type
        >::type
        on(Executor && exec) const
        {
            static_assert(
                hpx::traits::is_executor<Executor>::value ||
                hpx::traits::is_threads_executor<Executor>::value,
                "hpx::traits::is_executor<Executor>::value || "
                "hpx::traits::is_threads_executor<Executor>::value");

            typedef typename rebind_executor<
                dataseq_policy, Executor, executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor>(exec), parameters());
        }

        /// Create a new dataseq_policy from the given
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
        /// \returns The new dataseq_policy
        ///
        template <typename... Parameters, typename ParametersType =
            typename executor_parameters_join<Parameters...>::type>
        typename rebind_executor<
            dataseq_policy, executor_type, ParametersType
        >::type
        with(Parameters &&... params) const
        {
            typedef typename rebind_executor<
                dataseq_policy, executor_type, ParametersType
            >::type rebound_type;
            return rebound_type(executor(),
                join_executor_parameters(std::forward<Parameters>(params)...));
        }

    public:
        /// Return the associated executor object.
        /// Return the associated executor object.
        executor_type& executor() { return exec_; }
        /// Return the associated executor object.
        HPX_CONSTEXPR executor_type const& executor() const { return exec_; }

        /// Return the associated executor parameters object.
        executor_parameters_type& parameters() { return params_; }
        /// Return the associated executor parameters object.
        HPX_CONSTEXPR executor_parameters_type const& parameters() const
            { return params_; }

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
    HPX_STATIC_CONSTEXPR dataseq_policy dataseq;

    /// The class dataseq_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// require that a parallel algorithm's execution may not be parallelized.
    template <typename Executor, typename Parameters>
    struct dataseq_policy_shim : dataseq_policy
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
            typedef dataseq_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// Create a new dataseq_task_policy.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new dataseq_task_policy_shim
        ///
        HPX_CONSTEXPR dataseq_task_policy_shim<Executor, Parameters> operator()(
            task_policy_tag tag) const
        {
            return dataseq_task_policy_shim<
                Executor, Parameters>(exec_, params_);
        }

        /// Create a new dataseq_policy from the given
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
        /// \returns The new dataseq_policy
        ///
        template <typename Executor_>
        typename rebind_executor<
            dataseq_policy_shim, Executor_,
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
                dataseq_policy_shim, Executor_,
                executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor_>(exec), params_);
        }

        /// Create a new dataseq_policy_shim from the given
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
        /// \returns The new dataseq_policy_shim
        ///
        template <typename... Parameters_, typename ParametersType =
            typename executor_parameters_join<Parameters_...>::type>
        typename rebind_executor<
            dataseq_policy_shim, executor_type, ParametersType
        >::type
        with(Parameters_ &&... params) const
        {
            typedef typename rebind_executor<
                dataseq_policy_shim, executor_type, ParametersType
            >::type rebound_type;
            return rebound_type(exec_,
                join_executor_parameters(std::forward<Parameters_>(params)...));
        }

        /// Return the associated executor object.
        Executor& executor() { return exec_; }
        /// Return the associated executor object.
        HPX_CONSTEXPR Executor const& executor() const { return exec_; }

        /// Return the associated executor parameters object.
        Parameters& parameters() { return params_; }
        /// Return the associated executor parameters object.
        HPX_CONSTEXPR Parameters const& parameters() const { return params_; }

        /// \cond NOINTERNAL
        HPX_CONSTEXPR dataseq_policy_shim() {}

        template <typename Executor_, typename Parameters_>
        HPX_CONSTEXPR dataseq_policy_shim(
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
    /// Extension: The class datapar_task_policy is an execution
    /// policy type used as a unique type to disambiguate parallel algorithm
    /// overloading and indicate that a parallel algorithm's execution may be
    /// parallelized.
    ///
    /// The algorithm returns a future representing the result of the
    /// corresponding algorithm when invoked with the parallel_policy.
    struct datapar_task_policy
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
            typedef datapar_task_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// \cond NOINTERNAL
        HPX_CONSTEXPR datapar_task_policy() {}
        /// \endcond

        /// Create a new datapar_task_policy from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new datapar_task_policy
        ///
        HPX_CONSTEXPR datapar_task_policy operator()(task_policy_tag tag) const
        {
            return *this;
        }

        /// Create a new datapar_task_policy from given executor
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
        /// \returns The new datapar_task_policy
        ///
        template <typename Executor>
        typename rebind_executor<
            datapar_task_policy, Executor, executor_parameters_type
        >::type
        on(Executor && exec) const
        {
            static_assert(
                hpx::traits::is_executor<Executor>::value ||
                hpx::traits::is_threads_executor<Executor>::value,
                "hpx::traits::is_executor<Executor>::value || "
                "hpx::traits::is_threads_executor<Executor>::value");

            typedef typename rebind_executor<
                datapar_task_policy, Executor,
                executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor>(exec), parameters());
        }

        /// Create a new datapar_task_policy_shim from the given
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
        /// \returns The new datapar_policy_shim
        ///
        template <typename... Parameters, typename ParametersType =
            typename executor_parameters_join<Parameters...>::type>
        typename rebind_executor<
            datapar_task_policy, executor_type, ParametersType
        >::type
        with(Parameters &&... params) const
        {
            typedef typename rebind_executor<
                datapar_task_policy, executor_type, ParametersType
            >::type rebound_type;
            return rebound_type(executor(),
                join_executor_parameters(std::forward<Parameters>(params)...));
        }

    public:
        /// Return the associated executor object.
        executor_type& executor() { return exec_; }
        /// Return the associated executor object.
        HPX_CONSTEXPR executor_type const& executor() const { return exec_; }

        /// Return the associated executor parameters object.
        executor_parameters_type& parameters() { return params_; }
        /// Return the associated executor parameters object.
        HPX_CONSTEXPR executor_parameters_type const& parameters() const
            { return params_; }

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
    /// The class datapar_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// indicate that a parallel algorithm's execution may be parallelized.
    struct datapar_policy
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
            typedef datapar_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// \cond NOINTERNAL
        HPX_CONSTEXPR datapar_policy() : exec_{}, params_{} {}
        /// \endcond

        /// Create a new datapar_policy referencing a chunk size.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new datapar_task_policy
        ///
        HPX_CONSTEXPR datapar_task_policy operator()(task_policy_tag tag) const
        {
            return datapar_task_policy();
        }

        /// Create a new datapar_policy referencing an executor and
        /// a chunk size.
        ///
        /// \param exec         [in] The executor to use for the execution of
        ///                     the parallel algorithm the returned execution
        ///                     policy is used with
        ///
        /// \returns The new datapar_policy
        ///
        template <typename Executor>
        typename rebind_executor<
            datapar_policy, Executor, executor_parameters_type
        >::type
        on(Executor && exec) const
        {
            static_assert(
                hpx::traits::is_executor<Executor>::value ||
                hpx::traits::is_threads_executor<Executor>::value,
                "hpx::traits::is_executor<Executor>::value || "
                "hpx::traits::is_threads_executor<Executor>::value");

            typedef typename rebind_executor<
                datapar_policy, Executor, executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor>(exec), parameters());
        }

        /// Create a new datapar_policy from the given
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
        /// \returns The new datapar_policy
        ///
        template <typename... Parameters, typename ParametersType =
            typename executor_parameters_join<Parameters...>::type>
        typename rebind_executor<
            datapar_policy, executor_type, ParametersType
        >::type
        with(Parameters &&... params) const
        {
            typedef typename rebind_executor<
                datapar_policy, executor_type, ParametersType
            >::type rebound_type;
            return rebound_type(executor(),
                join_executor_parameters(std::forward<Parameters>(params)...));
        }

    public:
        /// Return the associated executor object.
        executor_type& executor() { return exec_; }
        /// Return the associated executor object.
        HPX_CONSTEXPR executor_type const& executor() const { return exec_; }

        /// Return the associated executor parameters object.
        executor_parameters_type& parameters() { return params_; }
        /// Return the associated executor parameters object.
        HPX_CONSTEXPR executor_parameters_type const& parameters() const
            { return params_; }

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
    HPX_STATIC_CONSTEXPR datapar_policy datapar;

    /// The class datapar_policy_shim is an execution policy type
    /// used as a unique type to disambiguate parallel algorithm overloading
    /// and indicate that a parallel algorithm's execution may be parallelized.
    template <typename Executor, typename Parameters>
    struct datapar_policy_shim : datapar_policy
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
            typedef datapar_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// Create a new datapar_task_policy_shim
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new datapar_task_policy_shim
        ///
        HPX_CONSTEXPR datapar_task_policy_shim<Executor, Parameters>
        operator()(task_policy_tag tag) const
        {
            return datapar_task_policy_shim<Executor, Parameters>(
                exec_, params_);
        }

        /// Create a new datapar_task_policy_shim from the given
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
        /// \returns The new parallel_policy
        ///
        template <typename Executor_>
        typename rebind_executor<
            datapar_policy_shim, Executor_,
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
                datapar_policy_shim, Executor_,
                executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor_>(exec), params_);
        }

        /// Create a new datapar_policy_shim from the given
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
        /// \returns The new datapar_policy_shim
        ///
        template <typename... Parameters_, typename ParametersType =
            typename executor_parameters_join<Parameters_...>::type>
        typename rebind_executor<
            datapar_policy_shim, executor_type, ParametersType
        >::type
        with(Parameters_ &&... params) const
        {
            typedef typename rebind_executor<
                datapar_policy_shim, executor_type, ParametersType
            >::type rebound_type;
            return rebound_type(exec_,
                join_executor_parameters(std::forward<Parameters_>(params)...));
        }

        /// Return the associated executor object.
        Executor& executor() { return exec_; }
        /// Return the associated executor object.
        HPX_CONSTEXPR Executor const& executor() const { return exec_; }

        /// Return the associated executor parameters object.
        Parameters& parameters() { return params_; }
        /// Return the associated executor parameters object.
        HPX_CONSTEXPR Parameters const& parameters() const { return params_; }

        /// \cond NOINTERNAL
        HPX_CONSTEXPR datapar_policy_shim() {}

        template <typename Executor_, typename Parameters_>
        HPX_CONSTEXPR datapar_policy_shim(
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

    /// The class datapar_task_policy_shim is an
    /// execution policy type used as a unique type to disambiguate parallel
    /// algorithm overloading based on combining a underlying
    /// \a datapar_task_policy and an executor and indicate that
    /// a parallel algorithm's execution may be parallelized and vectorized.
    template <typename Executor, typename Parameters>
    struct datapar_task_policy_shim : datapar_task_policy
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
            typedef datapar_task_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// Create a new datapar_task_policy_shim from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequenced_task_policy
        ///
        HPX_CONSTEXPR datapar_task_policy_shim operator()(
            task_policy_tag tag) const
        {
            return *this;
        }

        /// Create a new parallel_task_policy from the given
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
        /// \returns The new parallel_task_policy
        ///
        template <typename Executor_>
        typename rebind_executor<
            datapar_task_policy_shim, Executor_,
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
                datapar_task_policy_shim, Executor_,
                executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor_>(exec), params_);
        }

        /// Create a new parallel_policy_shim from the given
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
        /// \returns The new parallel_policy_shim
        ///
        template <typename... Parameters_, typename ParametersType =
            typename executor_parameters_join<Parameters_...>::type>
        typename rebind_executor<
            datapar_task_policy_shim, executor_type, ParametersType
        >::type
        with(Parameters_ &&... params) const
        {
            typedef typename rebind_executor<
                datapar_task_policy_shim, executor_type, ParametersType
            >::type rebound_type;
            return rebound_type(exec_,
                join_executor_parameters(std::forward<Parameters_>(params)...));
        }

        /// Return the associated executor object.
        Executor& executor() { return exec_; }
        /// Return the associated executor object.
        HPX_CONSTEXPR Executor const& executor() const { return exec_; }

        /// Return the associated executor parameters object.
        Parameters& parameters() { return params_; }
        /// Return the associated executor parameters object.
        HPX_CONSTEXPR Parameters const& parameters() const { return params_; }

        /// \cond NOINTERNAL
        HPX_CONSTEXPR datapar_task_policy_shim() {}

        template <typename Executor_, typename Parameters_>
        HPX_CONSTEXPR datapar_task_policy_shim(
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
}}}}

namespace hpx { namespace parallel { namespace execution
{
    ///////////////////////////////////////////////////////////////////////////
    // extensions
    namespace detail
    {
        /// \cond NOINTERNAL
        template <>
        struct is_execution_policy<dataseq_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_execution_policy<
                dataseq_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <>
        struct is_execution_policy<dataseq_task_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_execution_policy<
                dataseq_task_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <>
        struct is_execution_policy<datapar_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_execution_policy<
                datapar_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <>
        struct is_execution_policy<datapar_task_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_execution_policy<
                datapar_task_policy_shim<Executor, Parameters> >
          : std::true_type
        {};
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <>
        struct is_sequential_execution_policy<dataseq_policy>
          : std::true_type
        {};

        template <>
        struct is_sequential_execution_policy<dataseq_task_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_sequential_execution_policy<
                dataseq_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_sequential_execution_policy<
                dataseq_task_policy_shim<Executor, Parameters> >
          : std::true_type
        {};
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <>
        struct is_async_execution_policy<dataseq_task_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_async_execution_policy<
                dataseq_task_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <>
        struct is_async_execution_policy<datapar_task_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_async_execution_policy<
                datapar_task_policy_shim<Executor, Parameters> >
          : std::true_type
        {};
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <>
        struct is_parallel_execution_policy<datapar_policy>
          : std::true_type
        {};

        template <>
        struct is_parallel_execution_policy<datapar_task_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_parallel_execution_policy<
                datapar_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_parallel_execution_policy<
                datapar_task_policy_shim<Executor, Parameters> >
          : std::true_type
        {};
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <>
        struct is_vectorpack_execution_policy<dataseq_policy>
          : std::true_type
        {};

        template <>
        struct is_vectorpack_execution_policy<dataseq_task_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_vectorpack_execution_policy<
                dataseq_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_vectorpack_execution_policy<
                dataseq_task_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <>
        struct is_vectorpack_execution_policy<datapar_policy>
          : std::true_type
        {};

        template <>
        struct is_vectorpack_execution_policy<datapar_task_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_vectorpack_execution_policy<
                datapar_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_vectorpack_execution_policy<
                datapar_task_policy_shim<Executor, Parameters> >
          : std::true_type
        {};
        /// \endcond
    }
}}}

#if defined(HPX_HAVE_EXECUTION_POLICY_COMPATIBILITY)
///////////////////////////////////////////////////////////////////////////////
// Compatibility layer for changes introduced by C++17
namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    using dataseq_task_execution_policy = execution::dataseq_task_policy;
    using dataseq_execution_policy = execution::dataseq_policy;

    template <typename Executor, typename Parameters>
    using dataseq_task_execution_policy_shim =
        execution::dataseq_task_policy_shim<Executor, Parameters>;
    template <typename Executor, typename Parameters>
    using dataseq_execution_policy_shim =
        execution::dataseq_policy_shim<Executor, Parameters>;

    ///////////////////////////////////////////////////////////////////////////
    using datapar_task_execution_policy = execution::datapar_task_policy;
    using datapar_execution_policy = execution::datapar_policy;

    template <typename Executor, typename Parameters>
    using datapar_task_execution_policy_shim =
        execution::datapar_task_policy_shim<Executor, Parameters>;
    template <typename Executor, typename Parameters>
    using datapar_execution_policy_shim =
        execution::datapar_policy_shim<Executor, Parameters>;

    ///////////////////////////////////////////////////////////////////////////
    HPX_STATIC_CONSTEXPR dataseq_execution_policy dataseq_execution;
    HPX_STATIC_CONSTEXPR datapar_execution_policy datapar_execution;
}}}
#endif

#endif
#endif
