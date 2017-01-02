//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/execution_policy.hpp

#if !defined(HPX_PARALLEL_EXECUTION_POLICY_MAY_27_2014_0908PM)
#define HPX_PARALLEL_EXECUTION_POLICY_MAY_27_2014_0908PM

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/datapar/execution_policy.hpp>
#include <hpx/parallel/execution_policy_fwd.hpp>
#include <hpx/parallel/executors.hpp>
#include <hpx/parallel/executors/executor_parameters.hpp>
#include <hpx/parallel/executors/rebind_executor.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_execution_policy.hpp>
#include <hpx/traits/is_executor_v1.hpp>
#include <hpx/traits/is_executor_parameters.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/util/decay.hpp>

#include <memory>
#include <type_traits>
#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
#include <typeinfo>
#endif
#include <utility>

namespace hpx { namespace parallel { namespace execution
{
    ///////////////////////////////////////////////////////////////////////////
    /// Default sequential execution policy object.
    static task_policy_tag HPX_CONSTEXPR_OR_CONST task;

    ///////////////////////////////////////////////////////////////////////////
    /// Extension: The class sequenced_task_policy is an execution
    /// policy type used as a unique type to disambiguate parallel algorithm
    /// overloading and indicate that a parallel algorithm's execution may not
    /// be parallelized (has to run sequentially).
    ///
    /// The algorithm returns a future representing the result of the
    /// corresponding algorithm when invoked with the
    /// sequenced_policy.
    struct sequenced_task_policy
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
        typedef parallel::sequential_execution_tag execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef sequenced_task_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// \cond NOINTERNAL
        HPX_CONSTEXPR sequenced_task_policy() {}
        /// \endcond

        /// Create a new sequenced_task_policy from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequenced_task_policy
        ///
        HPX_CONSTEXPR sequenced_task_policy operator()(
            task_policy_tag tag) const
        {
            return *this;
        }

        /// Create a new sequenced_task_policy from the given
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
        /// \returns The new sequenced_task_policy
        ///
        template <typename Executor>
        typename rebind_executor<
            sequenced_task_policy, Executor,
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
                sequenced_task_policy, Executor,
                executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor>(exec), parameters());
        }

        /// Create a new sequenced_task_policy from the given
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
        /// \returns The new sequenced_task_policy
        ///
        template <typename... Parameters, typename ParametersType =
            typename executor_parameters_join<Parameters...>::type>
        typename rebind_executor<
            sequenced_task_policy, executor_type, ParametersType
        >::type
        with(Parameters &&... params) const
        {
            typedef typename rebind_executor<
                sequenced_task_policy, executor_type, ParametersType
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

    /// Extension: The class sequenced_task_policy_shim is an
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
    struct sequenced_task_policy_shim
      : sequenced_task_policy
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
            typedef sequenced_task_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// Create a new sequenced_task_policy from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequenced_task_policy
        ///
        HPX_CONSTEXPR sequenced_task_policy_shim const& operator()(
            task_policy_tag tag) const
        {
            return *this;
        }

        /// Create a new sequenced_task_policy from the given
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
        /// \returns The new sequenced_task_policy
        ///
        template <typename Executor_>
        typename rebind_executor<
            sequenced_task_policy_shim, Executor_,
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
                sequenced_task_policy_shim, Executor_,
                executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor_>(exec), params_);
        }

        /// Create a new sequenced_task_policy_shim from the given
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
            sequenced_task_policy_shim, executor_type, ParametersType
        >::type
        with(Parameters_ &&... params) const
        {
            typedef typename rebind_executor<
                sequenced_task_policy_shim, executor_type,
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
        HPX_CONSTEXPR sequenced_task_policy_shim() {}

        template <typename Executor_, typename Parameters_>
        HPX_CONSTEXPR sequenced_task_policy_shim(
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
    /// The class sequenced_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// require that a parallel algorithm's execution may not be parallelized.
    struct sequenced_policy
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
        typedef parallel::sequential_execution_tag execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef sequenced_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// \cond NOINTERNAL
        HPX_CONSTEXPR sequenced_policy() {}
        /// \endcond

        /// Create a new sequenced_task_policy.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequenced_task_policy
        ///
        HPX_CONSTEXPR sequenced_task_policy operator()(
            task_policy_tag tag) const
        {
            return sequenced_task_policy();
        }

        /// Create a new sequenced_policy from the given
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
        /// \returns The new sequenced_policy
        ///
        template <typename Executor>
        typename rebind_executor<
            sequenced_policy, Executor, executor_parameters_type
        >::type
        on(Executor && exec) const
        {
            static_assert(
                hpx::traits::is_executor<Executor>::value ||
                hpx::traits::is_threads_executor<Executor>::value,
                "hpx::traits::is_executor<Executor>::value || "
                "hpx::traits::is_threads_executor<Executor>::value");

            typedef typename rebind_executor<
                sequenced_policy, Executor, executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor>(exec), parameters());
        }

        /// Create a new sequenced_policy from the given
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
        /// \returns The new sequenced_policy
        ///
        template <typename... Parameters, typename ParametersType =
            typename executor_parameters_join<Parameters...>::type>
        typename rebind_executor<
            sequenced_policy, executor_type, ParametersType
        >::type
        with(Parameters &&... params) const
        {
            typedef typename rebind_executor<
                sequenced_policy, executor_type, ParametersType
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
    static sequenced_policy HPX_CONSTEXPR_OR_CONST seq;

    /// The class sequenced_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// require that a parallel algorithm's execution may not be parallelized.
    template <typename Executor, typename Parameters>
    struct sequenced_policy_shim : sequenced_policy
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
            typedef sequenced_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// Create a new sequenced_task_policy.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequenced_task_policy_shim
        ///
        HPX_CONSTEXPR sequenced_task_policy_shim<Executor, Parameters>
        operator()(task_policy_tag tag) const
        {
            return sequenced_task_policy_shim<
                Executor, Parameters>(exec_, params_);
        }

        /// Create a new sequenced_policy from the given
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
        /// \returns The new sequenced_policy
        ///
        template <typename Executor_>
        typename rebind_executor<
            sequenced_policy_shim, Executor_,
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
                sequenced_policy_shim, Executor_,
                executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor_>(exec), params_);
        }

        /// Create a new sequenced_policy_shim from the given
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
        /// \returns The new sequenced_policy_shim
        ///
        template <typename... Parameters_, typename ParametersType =
            typename executor_parameters_join<Parameters_...>::type>
        typename rebind_executor<
            sequenced_policy_shim, executor_type, ParametersType
        >::type
        with(Parameters_ &&... params) const
        {
            typedef typename rebind_executor<
                sequenced_policy_shim, executor_type, ParametersType
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
        HPX_CONSTEXPR sequenced_policy_shim() {}

        template <typename Executor_, typename Parameters_>
        HPX_CONSTEXPR sequenced_policy_shim(
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
    /// Extension: The class parallel_task_policy is an execution
    /// policy type used as a unique type to disambiguate parallel algorithm
    /// overloading and indicate that a parallel algorithm's execution may be
    /// parallelized.
    ///
    /// The algorithm returns a future representing the result of the
    /// corresponding algorithm when invoked with the parallel_policy.
    struct parallel_task_policy
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
        typedef parallel::parallel_execution_tag execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef parallel_task_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// \cond NOINTERNAL
        HPX_CONSTEXPR parallel_task_policy() {}
        /// \endcond

        /// Create a new parallel_task_policy from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new parallel_task_policy
        ///
        HPX_CONSTEXPR parallel_task_policy operator()(
            task_policy_tag tag) const
        {
            return *this;
        }

        /// Create a new parallel_task_policy from given executor
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
        template <typename Executor>
        typename rebind_executor<
            parallel_task_policy, Executor, executor_parameters_type
        >::type
        on(Executor && exec) const
        {
            static_assert(
                hpx::traits::is_executor<Executor>::value ||
                hpx::traits::is_threads_executor<Executor>::value,
                "hpx::traits::is_executor<Executor>::value || "
                "hpx::traits::is_threads_executor<Executor>::value");

            typedef typename rebind_executor<
                parallel_task_policy, Executor,
                executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor>(exec), parameters());
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
        template <typename... Parameters, typename ParametersType =
            typename executor_parameters_join<Parameters...>::type>
        typename rebind_executor<
            parallel_task_policy, executor_type, ParametersType
        >::type
        with(Parameters &&... params) const
        {
            typedef typename rebind_executor<
                parallel_task_policy, executor_type, ParametersType
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

    /// Extension: The class parallel_task_policy_shim is an
    /// execution policy type used as a unique type to disambiguate parallel
    /// algorithm overloading based on combining a underlying
    /// \a parallel_task_policy and an executor and indicate that
    /// a parallel algorithm's execution may be parallelized.
    template <typename Executor, typename Parameters>
    struct parallel_task_policy_shim : parallel_task_policy
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
            typedef parallel_task_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// Create a new parallel_task_policy_shim from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequenced_task_policy
        ///
        HPX_CONSTEXPR parallel_task_policy_shim operator()(
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
            parallel_task_policy_shim, Executor_,
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
                parallel_task_policy_shim, Executor_,
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
            parallel_task_policy_shim, executor_type, ParametersType
        >::type
        with(Parameters_ &&... params) const
        {
            typedef typename rebind_executor<
                parallel_task_policy_shim, executor_type, ParametersType
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
        HPX_CONSTEXPR parallel_task_policy_shim() {}

        template <typename Executor_, typename Parameters_>
        HPX_CONSTEXPR parallel_task_policy_shim(
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
    /// The class parallel_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// indicate that a parallel algorithm's execution may be parallelized.
    struct parallel_policy
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
        typedef parallel::parallel_execution_tag execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef parallel_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// \cond NOINTERNAL
        HPX_CONSTEXPR parallel_policy() {}
        /// \endcond

        /// Create a new parallel_policy referencing a chunk size.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new parallel_policy
        ///
        HPX_CONSTEXPR parallel_task_policy operator()(
            task_policy_tag tag) const
        {
            return parallel_task_policy();
        }

        /// Create a new parallel_policy referencing an executor and
        /// a chunk size.
        ///
        /// \param exec         [in] The executor to use for the execution of
        ///                     the parallel algorithm the returned execution
        ///                     policy is used with
        ///
        /// \returns The new parallel_policy
        ///
        template <typename Executor>
        typename rebind_executor<
            parallel_policy, Executor, executor_parameters_type
        >::type
        on(Executor && exec) const
        {
            static_assert(
                hpx::traits::is_executor<Executor>::value ||
                hpx::traits::is_threads_executor<Executor>::value,
                "hpx::traits::is_executor<Executor>::value || "
                "hpx::traits::is_threads_executor<Executor>::value");

            typedef typename rebind_executor<
                parallel_policy, Executor, executor_parameters_type
            >::type rebound_type;
            return rebound_type(std::forward<Executor>(exec), parameters());
        }

        /// Create a new parallel_policy from the given
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
        /// \returns The new parallel_policy
        ///
        template <typename... Parameters, typename ParametersType =
            typename executor_parameters_join<Parameters...>::type>
        typename rebind_executor<
            parallel_policy, executor_type, ParametersType
        >::type
        with(Parameters &&... params) const
        {
            typedef typename rebind_executor<
                parallel_policy, executor_type, ParametersType
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

    /// Default parallel execution policy object.
    static parallel_policy HPX_CONSTEXPR_OR_CONST par;

    /// The class parallel_policy_shim is an execution policy type
    /// used as a unique type to disambiguate parallel algorithm overloading
    /// and indicate that a parallel algorithm's execution may be parallelized.
    template <typename Executor, typename Parameters>
    struct parallel_policy_shim : parallel_policy
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
            typedef parallel_policy_shim<
                    Executor_, Parameters_
                > type;
        };

        /// Create a new parallel_policy referencing a chunk size.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new parallel_policy
        ///
        HPX_CONSTEXPR parallel_task_policy_shim<Executor, Parameters>
        operator()(task_policy_tag tag) const
        {
            return parallel_task_policy_shim<Executor, Parameters>(
                exec_, params_);
        }

        /// Create a new parallel_policy from the given
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
            parallel_policy_shim, Executor_,
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
                parallel_policy_shim, Executor_,
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
        /// \note Requires: is_executor_parameters<Parameters>::value is true
        ///
        /// \returns The new parallel_policy_shim
        ///
        template <typename... Parameters_, typename ParametersType =
            typename executor_parameters_join<Parameters_...>::type>
        typename rebind_executor<
            parallel_policy_shim, executor_type, ParametersType
        >::type
        with(Parameters_ &&... params) const
        {
            typedef typename rebind_executor<
                parallel_policy_shim, executor_type, ParametersType
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
        HPX_CONSTEXPR parallel_policy_shim() {}

        template <typename Executor_, typename Parameters_>
        HPX_CONSTEXPR parallel_policy_shim(
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
    /// The class parallel_unsequenced_policy is an execution policy type
    /// used as a unique type to disambiguate parallel algorithm overloading
    /// and indicate that a parallel algorithm's execution may be vectorized.
    struct parallel_unsequenced_policy
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
        typedef parallel::parallel_execution_tag execution_category;

        /// \cond NOINTERNAL
        HPX_CONSTEXPR parallel_unsequenced_policy() {}
        /// \endcond

        /// Create a new parallel_unsequenced_policy from itself
        ///
        /// \param tag [in] Specify that the corresponding asynchronous
        ///            execution policy should be used
        ///
        /// \returns The new parallel_unsequenced_policy
        ///
        parallel_unsequenced_policy operator()(
            task_policy_tag tag) const
        {
            return *this;
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

    /// Default vector execution policy object.
    static parallel_unsequenced_policy HPX_CONSTEXPR_OR_CONST par_unseq;

    ///////////////////////////////////////////////////////////////////////////
    // Allow to detect execution policies which were created as a result
    // of a rebind operation. This information can be used to inhibit the
    // construction of a generic execution_policy from any of the rebound
    // policies.
    namespace detail
    {
        template <typename Executor, typename Parameters>
        struct is_rebound_execution_policy<
                sequenced_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_rebound_execution_policy<
                sequenced_task_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_rebound_execution_policy<
                parallel_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_rebound_execution_policy<
                parallel_task_policy_shim<Executor, Parameters> >
          : std::true_type
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <>
        struct is_execution_policy<parallel_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_execution_policy<
                parallel_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <>
        struct is_execution_policy<parallel_unsequenced_policy>
          : std::true_type
        {};

        template <>
        struct is_execution_policy<sequenced_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_execution_policy<
                sequenced_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        // extension
        template <>
        struct is_execution_policy<sequenced_task_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_execution_policy<
                sequenced_task_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <>
        struct is_execution_policy<parallel_task_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_execution_policy<
                parallel_task_policy_shim<Executor, Parameters> >
          : std::true_type
        {};
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <>
        struct is_parallel_execution_policy<parallel_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_parallel_execution_policy<
                parallel_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <>
        struct is_parallel_execution_policy<
                parallel_unsequenced_policy>
          : std::true_type
        {};

        template <>
        struct is_parallel_execution_policy<parallel_task_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_parallel_execution_policy<
                parallel_task_policy_shim<Executor, Parameters> >
          : std::true_type
        {};
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <>
        struct is_sequential_execution_policy<sequenced_task_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_sequential_execution_policy<
                sequenced_task_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <>
        struct is_sequential_execution_policy<sequenced_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_sequential_execution_policy<
                sequenced_policy_shim<Executor, Parameters> >
          : std::true_type
        {};
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <>
        struct is_async_execution_policy<sequenced_task_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_async_execution_policy<
                sequenced_task_policy_shim<Executor, Parameters> >
          : std::true_type
        {};

        template <>
        struct is_async_execution_policy<parallel_task_policy>
          : std::true_type
        {};

        template <typename Executor, typename Parameters>
        struct is_async_execution_policy<
                parallel_task_policy_shim<Executor, Parameters> >
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
    using sequential_task_execution_policy =
        parallel::execution::sequenced_task_policy;
    using sequential_execution_policy =
        parallel::execution::sequenced_policy;

    template <typename Executor, typename Parameters>
    using sequential_task_execution_policy_shim =
        parallel::execution::sequenced_task_policy_shim<Executor, Parameters>;
    template <typename Executor, typename Parameters>
    using sequential_execution_policy_shim =
        parallel::execution::sequenced_policy_shim<Executor, Parameters>;

    ///////////////////////////////////////////////////////////////////////////
    using parallel_task_execution_policy =
        parallel::execution::parallel_task_policy;
    using parallel_execution_policy =
        parallel::execution::parallel_policy;

    template <typename Executor, typename Parameters>
    using parallel_task_execution_policy_shim =
        parallel::execution::parallel_task_policy_shim<Executor, Parameters>;
    template <typename Executor, typename Parameters>
    using parallel_execution_policy_shim =
        parallel::execution::parallel_policy_shim<Executor, Parameters>;

    ///////////////////////////////////////////////////////////////////////////
    using parallel_vector_execution_policy =
        parallel::execution::parallel_unsequenced_policy;

    using task_execution_policy_tag = parallel::execution::task_policy_tag;

    ///////////////////////////////////////////////////////////////////////////
    static sequential_execution_policy HPX_CONSTEXPR_OR_CONST seq;
    static parallel_execution_policy HPX_CONSTEXPR_OR_CONST par;
    static parallel_vector_execution_policy HPX_CONSTEXPR_OR_CONST par_vec;

    static task_execution_policy_tag HPX_CONSTEXPR_OR_CONST task;

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    ///////////////////////////////////////////////////////////////////////////
    class execution_policy;

    namespace detail
    {
        /// \cond NOINTERNAL
        struct execution_policy_base
        {
            virtual ~execution_policy_base() {}

            virtual std::type_info const& type() const HPX_NOEXCEPT  = 0;

            virtual execution_policy make_async(
                task_policy_tag tag) const = 0;
            virtual launch launch_policy() const = 0;

            virtual void* get() HPX_NOEXCEPT = 0;
            virtual void const* get() const HPX_NOEXCEPT = 0;
        };

        template <typename ExPolicy>
        struct execution_policy_shim : execution_policy_base
        {
            execution_policy_shim(ExPolicy const& policy)
              : policy_(policy)
            {}
            execution_policy_shim(ExPolicy && policy)
              : policy_(std::move(policy))
            {}

            std::type_info const& type() const HPX_NOEXCEPT
            {
                return typeid(ExPolicy);
            }

            // defined below
            execution_policy make_async(task_policy_tag tag) const;
            launch launch_policy() const;

            void* get() HPX_NOEXCEPT
            {
                return &policy_;
            }

            void const* get() const HPX_NOEXCEPT
            {
                return &policy_;
            }

        private:
            ExPolicy policy_;
        };

        template <>
        struct is_execution_policy<execution_policy>
          : std::true_type
        {};
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    ///
    /// An execution policy is an object that expresses the requirements on the
    /// ordering of functions invoked as a consequence of the invocation of a
    /// standard algorithm. Execution policies afford standard algorithms the
    /// discretion to execute in parallel.
    ///
    /// 1. The class execution_policy is a dynamic container for execution
    ///    policy objects. execution_policy allows dynamic control over
    ///    standard algorithm execution.
    /// 2. Objects of type execution_policy shall be constructible and
    ///    assignable from objects of type T for which
    ///    is_execution_policy<T>::value is true.
    ///
    class execution_policy
    {
    private:
        std::shared_ptr<detail::execution_policy_base> inner_;

    public:
        /// Effects: Constructs an execution_policy object with a copy of
        ///          exec's state
        /// Requires: is_execution_policy<T>::value is true
        ///
        /// \param policy Specifies the inner execution policy
        template <typename ExPolicy>
        execution_policy(ExPolicy const& policy,
                typename std::enable_if<
                    execution::is_execution_policy<ExPolicy>::value &&
                        !is_rebound_execution_policy<ExPolicy>::value,
                    ExPolicy
                >::type* = nullptr)
          : inner_(std::make_shared<
                    detail::execution_policy_shim<ExPolicy>
                >(policy))
        {}

        /// Move constructs a new execution_policy object.
        ///
        /// \param policy Specifies the inner execution policy
        execution_policy(execution_policy && policy)
          : inner_(std::move(policy.inner_))
        {}

        /// Copy constructs a new execution_policy object.
        ///
        /// \param rhs Specifies the inner execution policy
        execution_policy(execution_policy const& rhs)
          : inner_(rhs.inner_)
        {}

        /// Extension: Create a new execution_policy holding the current policy
        /// made asynchronous.
        ///
        /// \param tag  [in] Specify that the corresponding asynchronous
        ///             execution policy should be used
        ///
        /// \returns The new execution_policy
        ///
        execution_policy operator()(task_policy_tag tag) const
        {
            return inner_->make_async(tag);
        }

        /// Extension: Retrieve default launch policy for this execution policy.
        ///
        /// \returns The associated default launch policy
        ///
        launch launch_policy() const
        {
            return inner_->launch_policy();
        }

        /// Effects: Assigns a copy of exec's state to *this
        /// Returns: *this
        /// Requires: is_execution_policy<T>::value is true
        ///
        /// \param policy Specifies the inner execution policy
        template <typename ExPolicy>
        typename std::enable_if<
            execution::is_execution_policy<ExPolicy>::value &&
                !is_rebound_execution_policy<ExPolicy>::value,
            execution_policy
        >::type&
        operator=(ExPolicy const& policy)
        {
            if (this != &policy)
            {
                inner_ = std::make_shared<
                        detail::execution_policy_shim<ExPolicy>
                    >(policy);
            }
            return *this;
        }

        /// Move assigns a new execution policy to the object.
        ///
        /// \param policy Specifies the inner execution policy
        execution_policy& operator=(execution_policy && policy)
        {
            if (this != &policy)
                inner_ = std::move(policy.inner_);
            return *this;
        }

        /// Returns: typeid(T), such that T is the type of the execution policy
        ///          object contained by *this
        std::type_info const& type() const HPX_NOEXCEPT
        {
            return inner_->type();
        }

        /// Returns: If target_type() == typeid(T), a pointer to the stored
        ///          execution policy object; otherwise a null pointer
        /// Requires: is_execution_policy<T>::value is true
        template <typename ExPolicy>
        ExPolicy* get() HPX_NOEXCEPT
        {
            static_assert(
                !(std::is_same<ExPolicy, execution_policy>::value),
                "Incorrect execution policy parameter.");
            static_assert(
                execution::is_execution_policy<ExPolicy>::value,
                "Execution policy type required.");

            if (inner_->type() != typeid(ExPolicy))
                return 0;

            return static_cast<ExPolicy*>(inner_->get());
        }

        /// Returns: If target_type() == typeid(T), a pointer to the stored
        ///          execution policy object; otherwise a null pointer
        /// Requires: is_execution_policy<T>::value is true
        template <typename ExPolicy>
        ExPolicy const* get() const HPX_NOEXCEPT
        {
            static_assert(
                !(std::is_same<ExPolicy, execution_policy>::value),
                "Incorrect execution policy parameter.");
            static_assert(
                execution::is_execution_policy<ExPolicy>::value,
                "Execution policy type required.");

            if (inner_->type() != typeid(ExPolicy))
                return 0;

            return static_cast<ExPolicy const*>(inner_->get());
        }
    };

    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename ExPolicy>
        execution_policy execution_policy_shim<ExPolicy>::make_async(
            task_policy_tag tag) const
        {
            return policy_(tag);
        }

        template <typename ExPolicy, typename Enable = void>
        struct extract_launch_policy
        {
            static launch call() { return launch::async; }
        };

        template <typename ExPolicy>
        struct extract_launch_policy<ExPolicy,
            typename std::enable_if<
                is_sequential_execution_policy<ExPolicy>::value
            >::type>
        {
            static launch call() { return launch::deferred; }
        };

        template <typename ExPolicy>
        launch
        execution_policy_shim<ExPolicy>::launch_policy() const
        {
            return extract_launch_policy<ExPolicy>::call();
        }
        /// \endcond
    }
#endif
}}}
#endif

#endif
