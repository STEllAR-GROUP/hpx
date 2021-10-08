//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution/executors/rebind_executor.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/executors/datapar/execution_policy_fwd.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/executors/sequenced_executor.hpp>
#include <hpx/serialization/serialize.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace execution { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    /// Extension: The class simd_task_policy is an execution
    /// policy type used as a unique type to disambiguate parallel algorithm
    /// overloading and indicate that a parallel algorithm's execution may not
    /// be parallelized (has to run sequentially).
    ///
    /// The algorithm returns a future representing the result of the
    /// corresponding algorithm when invoked with the
    /// sequenced_policy.
    struct simd_task_policy
    {
        /// The type of the executor associated with this execution policy
        typedef sequenced_executor executor_type;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        typedef parallel::execution::extract_executor_parameters<
            executor_type>::type executor_parameters_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef unsequenced_execution_tag execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef simd_task_policy_shim<Executor_, Parameters_> type;
        };

        /// \cond NOINTERNAL
        constexpr simd_task_policy() = default;
        /// \endcond

        /// Create a new simd_task_policy from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequenced_task_policy
        ///
        constexpr simd_task_policy operator()(task_policy_tag) const
        {
            return *this;
        }

        /// Create a new non task policy from itself
        ///
        /// \returns The non task simd policy
        ///
        inline decltype(auto) operator()(non_task_policy_tag /*tag*/) const;

        /// Create a new simd_task_policy from the given
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
        /// \returns The new simd_task_policy
        ///
        template <typename Executor>
        typename parallel::execution::rebind_executor<simd_task_policy,
            Executor, executor_parameters_type>::type
        on(Executor&& exec) const
        {
            static_assert(hpx::traits::is_executor_any<Executor>::value,
                "hpx::traits::is_executor_any<Executor>::value");

            typedef
                typename parallel::execution::rebind_executor<simd_task_policy,
                    Executor, executor_parameters_type>::type rebound_type;
            return rebound_type(std::forward<Executor>(exec), parameters());
        }

        /// Create a new simd_task_policy from the given
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
        /// \returns The new simd_task_policy
        ///
        template <typename... Parameters,
            typename ParametersType = typename parallel::execution::
                executor_parameters_join<Parameters...>::type>
        typename parallel::execution::rebind_executor<simd_task_policy,
            executor_type, ParametersType>::type
        with(Parameters&&... params) const
        {
            typedef
                typename parallel::execution::rebind_executor<simd_task_policy,
                    executor_type, ParametersType>::type rebound_type;
            return rebound_type(executor(),
                parallel::execution::join_executor_parameters(
                    std::forward<Parameters>(params)...));
        }

    public:
        /// Return the associated executor object.
        executor_type& executor()
        {
            return exec_;
        }
        /// Return the associated executor object.
        constexpr executor_type const& executor() const
        {
            return exec_;
        }

        /// Return the associated executor parameters object.
        executor_parameters_type& parameters()
        {
            return params_;
        }
        /// Return the associated executor parameters object.
        constexpr executor_parameters_type const& parameters() const
        {
            return params_;
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        constexpr void serialize(Archive&, const unsigned int)
        {
        }

    private:
        executor_type exec_;
        executor_parameters_type params_;
    };

    /// Extension: The class simd_task_policy_shim is an
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
    struct simd_task_policy_shim : simd_task_policy
    {
        /// The type of the executor associated with this execution policy
        typedef Executor executor_type;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        typedef Parameters executor_parameters_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef typename hpx::traits::executor_execution_category<
            executor_type>::type execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef simd_task_policy_shim<Executor_, Parameters_> type;
        };

        /// Create a new simd_task_policy_shim from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequenced_task_policy
        ///
        constexpr simd_task_policy_shim const& operator()(task_policy_tag) const
        {
            return *this;
        }

        /// Create a new non task policy from itself
        ///
        /// \returns The non task simd_policy_shim
        ///
        inline decltype(auto) operator()(non_task_policy_tag /*tag*/) const;

        /// Create a new simd_task_policy_shim from the given
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
        /// \returns The new simd_task_policy_shim
        ///
        template <typename Executor_>
        typename parallel::execution::rebind_executor<simd_task_policy_shim,
            Executor_, executor_parameters_type>::type
        on(Executor_&& exec) const
        {
            static_assert(hpx::traits::is_executor_any<Executor_>::value,
                "hpx::traits::is_executor_any<Executor_>::value");

            typedef typename parallel::execution::rebind_executor<
                simd_task_policy_shim, Executor_,
                executor_parameters_type>::type rebound_type;
            return rebound_type(std::forward<Executor_>(exec), params_);
        }

        /// Create a new simd_task_policy_shim from the given
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
        template <typename... Parameters_,
            typename ParametersType = typename parallel::execution::
                executor_parameters_join<Parameters_...>::type>
        typename parallel::execution::rebind_executor<simd_task_policy_shim,
            executor_type, ParametersType>::type
        with(Parameters_&&... params) const
        {
            typedef typename parallel::execution::rebind_executor<
                simd_task_policy_shim, executor_type, ParametersType>::type
                rebound_type;
            return rebound_type(exec_,
                parallel::execution::join_executor_parameters(
                    std::forward<Parameters_>(params)...));
        }

        /// Return the associated executor object.
        Executor& executor()
        {
            return exec_;
        }
        /// Return the associated executor object.
        constexpr Executor const& executor() const
        {
            return exec_;
        }

        /// Return the associated executor parameters object.
        Parameters& parameters()
        {
            return params_;
        }
        /// Return the associated executor parameters object.
        constexpr Parameters const& parameters() const
        {
            return params_;
        }

        /// \cond NOINTERNAL
        constexpr simd_task_policy_shim() = default;

        template <typename Executor_, typename Parameters_>
        constexpr simd_task_policy_shim(Executor_&& exec, Parameters_&& params)
          : exec_(std::forward<Executor_>(exec))
          , params_(std::forward<Parameters_>(params))
        {
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int /* version */)
        {
            // clang-format off
            ar & exec_ & params_;
            // clang-format on
        }

    private:
        Executor exec_;
        Parameters params_;
        /// \endcond
    };

    ///////////////////////////////////////////////////////////////////////////
    /// The class simd_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// require that a parallel algorithm's execution may not be parallelized.
    struct simd_policy
    {
        /// The type of the executor associated with this execution policy
        typedef sequenced_executor executor_type;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        typedef parallel::execution::extract_executor_parameters<
            executor_type>::type executor_parameters_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef unsequenced_execution_tag execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef simd_policy_shim<Executor_, Parameters_> type;
        };

        /// \cond NOINTERNAL
        constexpr simd_policy() = default;
        /// \endcond

        /// Create a new simd_task_policy.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new simd_task_policy
        ///
        constexpr simd_task_policy operator()(task_policy_tag) const
        {
            return simd_task_policy();
        }

        /// Create a new non task policy from itself
        ///
        /// \returns The non task simd policy
        ///
        constexpr decltype(auto) operator()(non_task_policy_tag /*tag*/) const
        {
            return *this;
        }

        /// Create a new simd_policy from the given
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
        /// \returns The new simd_policy
        ///
        template <typename Executor>
        typename parallel::execution::rebind_executor<simd_policy, Executor,
            executor_parameters_type>::type
        on(Executor&& exec) const
        {
            static_assert(hpx::traits::is_executor_any<Executor>::value,
                "hpx::traits::is_executor_any<Executor>::value");

            typedef typename parallel::execution::rebind_executor<simd_policy,
                Executor, executor_parameters_type>::type rebound_type;
            return rebound_type(std::forward<Executor>(exec), parameters());
        }

        /// Create a new simd_policy from the given
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
        /// \returns The new simd_policy
        ///
        template <typename... Parameters,
            typename ParametersType = typename parallel::execution::
                executor_parameters_join<Parameters...>::type>
        typename parallel::execution::rebind_executor<simd_policy,
            executor_type, ParametersType>::type
        with(Parameters&&... params) const
        {
            typedef typename parallel::execution::rebind_executor<simd_policy,
                executor_type, ParametersType>::type rebound_type;
            return rebound_type(executor(),
                parallel::execution::join_executor_parameters(
                    std::forward<Parameters>(params)...));
        }

    public:
        /// Return the associated executor object.
        /// Return the associated executor object.
        executor_type& executor()
        {
            return exec_;
        }
        /// Return the associated executor object.
        constexpr executor_type const& executor() const
        {
            return exec_;
        }

        /// Return the associated executor parameters object.
        executor_parameters_type& parameters()
        {
            return params_;
        }
        /// Return the associated executor parameters object.
        constexpr executor_parameters_type const& parameters() const
        {
            return params_;
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& /* ar */, const unsigned int /* version */)
        {
        }

    private:
        executor_type exec_;
        executor_parameters_type params_;
    };

    /// Default sequential execution policy object.
    static constexpr simd_policy simd{};

    /// The class simd_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// require that a parallel algorithm's execution may not be parallelized.
    template <typename Executor, typename Parameters>
    struct simd_policy_shim : simd_policy
    {
        /// The type of the executor associated with this execution policy
        typedef Executor executor_type;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        typedef Parameters executor_parameters_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef typename hpx::traits::executor_execution_category<
            executor_type>::type execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef simd_policy_shim<Executor_, Parameters_> type;
        };

        /// Create a new simd_task_policy.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new simd_task_policy_shim
        ///
        constexpr simd_task_policy_shim<Executor, Parameters> operator()(
            task_policy_tag) const
        {
            return simd_task_policy_shim<Executor, Parameters>(exec_, params_);
        }

        /// Create a new non task policy from itself
        ///
        /// \returns The non task simd_policy_shim
        ///
        constexpr decltype(auto) operator()(non_task_policy_tag /*tag*/) const
        {
            return *this;
        }

        /// Create a new simd_policy from the given
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
        /// \returns The new simd_policy
        ///
        template <typename Executor_>
        typename parallel::execution::rebind_executor<simd_policy_shim,
            Executor_, executor_parameters_type>::type
        on(Executor_&& exec) const
        {
            static_assert(hpx::traits::is_executor_any<Executor_>::value,
                "hpx::traits::is_executor_any<Executor_>::value");

            typedef
                typename parallel::execution::rebind_executor<simd_policy_shim,
                    Executor_, executor_parameters_type>::type rebound_type;
            return rebound_type(std::forward<Executor_>(exec), params_);
        }

        /// Create a new simd_policy_shim from the given
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
        /// \returns The new simd_policy_shim
        ///
        template <typename... Parameters_,
            typename ParametersType = typename parallel::execution::
                executor_parameters_join<Parameters_...>::type>
        typename parallel::execution::rebind_executor<simd_policy_shim,
            executor_type, ParametersType>::type
        with(Parameters_&&... params) const
        {
            typedef
                typename parallel::execution::rebind_executor<simd_policy_shim,
                    executor_type, ParametersType>::type rebound_type;
            return rebound_type(exec_,
                parallel::execution::join_executor_parameters(
                    std::forward<Parameters_>(params)...));
        }

        /// Return the associated executor object.
        Executor& executor()
        {
            return exec_;
        }
        /// Return the associated executor object.
        constexpr Executor const& executor() const
        {
            return exec_;
        }

        /// Return the associated executor parameters object.
        Parameters& parameters()
        {
            return params_;
        }
        /// Return the associated executor parameters object.
        constexpr Parameters const& parameters() const
        {
            return params_;
        }

        /// \cond NOINTERNAL
        constexpr simd_policy_shim() = default;

        template <typename Executor_, typename Parameters_>
        constexpr simd_policy_shim(Executor_&& exec, Parameters_&& params)
          : exec_(std::forward<Executor_>(exec))
          , params_(std::forward<Parameters_>(params))
        {
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int /* version */)
        {
            // clang-format off
            ar & exec_ & params_;
            // clang-format on
        }

    private:
        Executor exec_;
        Parameters params_;
        /// \endcond
    };

    ///////////////////////////////////////////////////////////////////////////
    /// Extension: The class par_simd_task_policy is an execution
    /// policy type used as a unique type to disambiguate parallel algorithm
    /// overloading and indicate that a parallel algorithm's execution may be
    /// parallelized.
    ///
    /// The algorithm returns a future representing the result of the
    /// corresponding algorithm when invoked with the parallel_policy.
    struct par_simd_task_policy
    {
        /// The type of the executor associated with this execution policy
        typedef parallel_executor executor_type;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        typedef parallel::execution::extract_executor_parameters<
            executor_type>::type executor_parameters_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef unsequenced_execution_tag execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef par_simd_task_policy_shim<Executor_, Parameters_> type;
        };

        /// \cond NOINTERNAL
        constexpr par_simd_task_policy() = default;
        /// \endcond

        /// Create a new par_simd_task_policy from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new par_simd_task_policy
        ///
        constexpr par_simd_task_policy operator()(task_policy_tag) const
        {
            return *this;
        }

        /// Create a new non task policy from itself
        ///
        /// \returns The non task par_simd policy
        ///
        inline decltype(auto) operator()(non_task_policy_tag /*tag*/) const;

        /// Create a new par_simd_task_policy from given executor
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
        /// \returns The new par_simd_task_policy
        ///
        template <typename Executor>
        typename parallel::execution::rebind_executor<par_simd_task_policy,
            Executor, executor_parameters_type>::type
        on(Executor&& exec) const
        {
            static_assert(hpx::traits::is_executor_any<Executor>::value,
                "hpx::traits::is_executor_any<Executor>::value");

            typedef typename parallel::execution::rebind_executor<
                par_simd_task_policy, Executor, executor_parameters_type>::type
                rebound_type;
            return rebound_type(std::forward<Executor>(exec), parameters());
        }

        /// Create a new par_simd_task_policy_shim from the given
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
        /// \returns The new par_simd_policy_shim
        ///
        template <typename... Parameters,
            typename ParametersType = typename parallel::execution::
                executor_parameters_join<Parameters...>::type>
        typename parallel::execution::rebind_executor<par_simd_task_policy,
            executor_type, ParametersType>::type
        with(Parameters&&... params) const
        {
            typedef typename parallel::execution::rebind_executor<
                par_simd_task_policy, executor_type, ParametersType>::type
                rebound_type;
            return rebound_type(executor(),
                parallel::execution::join_executor_parameters(
                    std::forward<Parameters>(params)...));
        }

    public:
        /// Return the associated executor object.
        executor_type& executor()
        {
            return exec_;
        }
        /// Return the associated executor object.
        constexpr executor_type const& executor() const
        {
            return exec_;
        }

        /// Return the associated executor parameters object.
        executor_parameters_type& parameters()
        {
            return params_;
        }
        /// Return the associated executor parameters object.
        constexpr executor_parameters_type const& parameters() const
        {
            return params_;
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& /* ar */, const unsigned int /* version */)
        {
        }

    private:
        executor_type exec_;
        executor_parameters_type params_;
    };

    ///////////////////////////////////////////////////////////////////////////
    /// The class par_simd_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// indicate that a parallel algorithm's execution may be parallelized.
    struct par_simd_policy
    {
        /// The type of the executor associated with this execution policy
        typedef parallel_executor executor_type;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        typedef parallel::execution::extract_executor_parameters<
            executor_type>::type executor_parameters_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef unsequenced_execution_tag execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef par_simd_policy_shim<Executor_, Parameters_> type;
        };

        /// \cond NOINTERNAL
        constexpr par_simd_policy() = default;
        /// \endcond

        /// Create a new par_simd_policy referencing a chunk size.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new par_simd_task_policy
        ///
        constexpr par_simd_task_policy operator()(task_policy_tag) const
        {
            return par_simd_task_policy();
        }

        /// Create a new non task policy from itself
        ///
        /// \returns The non task par_simd policy
        ///
        constexpr decltype(auto) operator()(non_task_policy_tag /*tag*/) const
        {
            return *this;
        }

        /// Create a new par_simd_policy referencing an executor and
        /// a chunk size.
        ///
        /// \param exec         [in] The executor to use for the execution of
        ///                     the parallel algorithm the returned execution
        ///                     policy is used with
        ///
        /// \returns The new par_simd_policy
        ///
        template <typename Executor>
        typename parallel::execution::rebind_executor<par_simd_policy, Executor,
            executor_parameters_type>::type
        on(Executor&& exec) const
        {
            static_assert(hpx::traits::is_executor_any<Executor>::value,
                "hpx::traits::is_executor_any<Executor>::value");

            typedef
                typename parallel::execution::rebind_executor<par_simd_policy,
                    Executor, executor_parameters_type>::type rebound_type;
            return rebound_type(std::forward<Executor>(exec), parameters());
        }

        /// Create a new par_simd_policy from the given
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
        /// \returns The new par_simd_policy
        ///
        template <typename... Parameters,
            typename ParametersType = typename parallel::execution::
                executor_parameters_join<Parameters...>::type>
        typename parallel::execution::rebind_executor<par_simd_policy,
            executor_type, ParametersType>::type
        with(Parameters&&... params) const
        {
            typedef
                typename parallel::execution::rebind_executor<par_simd_policy,
                    executor_type, ParametersType>::type rebound_type;
            return rebound_type(executor(),
                parallel::execution::join_executor_parameters(
                    std::forward<Parameters>(params)...));
        }

    public:
        /// Return the associated executor object.
        executor_type& executor()
        {
            return exec_;
        }
        /// Return the associated executor object.
        constexpr executor_type const& executor() const
        {
            return exec_;
        }

        /// Return the associated executor parameters object.
        executor_parameters_type& parameters()
        {
            return params_;
        }
        /// Return the associated executor parameters object.
        constexpr executor_parameters_type const& parameters() const
        {
            return params_;
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& /* ar */, const unsigned int /* version */)
        {
        }

    private:
        executor_type exec_;
        executor_parameters_type params_;
    };

    /// Default data-parallel execution policy object.
    static constexpr par_simd_policy par_simd{};

    /// The class par_simd_policy_shim is an execution policy type
    /// used as a unique type to disambiguate parallel algorithm overloading
    /// and indicate that a parallel algorithm's execution may be parallelized.
    template <typename Executor, typename Parameters>
    struct par_simd_policy_shim : par_simd_policy
    {
        /// The type of the executor associated with this execution policy
        typedef Executor executor_type;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        typedef Parameters executor_parameters_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef typename hpx::traits::executor_execution_category<
            executor_type>::type execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef par_simd_policy_shim<Executor_, Parameters_> type;
        };

        /// Create a new par_simd_task_policy_shim
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new par_simd_task_policy_shim
        ///
        constexpr par_simd_task_policy_shim<Executor, Parameters> operator()(
            task_policy_tag) const
        {
            return par_simd_task_policy_shim<Executor, Parameters>(
                exec_, params_);
        }

        /// Create a new non task policy from itself
        ///
        /// \returns The non task par_simd policy
        ///
        constexpr decltype(auto) operator()(non_task_policy_tag /*tag*/) const
        {
            return *this;
        }

        /// Create a new par_simd_task_policy_shim from the given
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
        typename parallel::execution::rebind_executor<par_simd_policy_shim,
            Executor_, executor_parameters_type>::type
        on(Executor_&& exec) const
        {
            static_assert(hpx::traits::is_executor_any<Executor_>::value,
                "hpx::traits::is_executor_any<Executor_>::value");

            typedef typename parallel::execution::rebind_executor<
                par_simd_policy_shim, Executor_, executor_parameters_type>::type
                rebound_type;
            return rebound_type(std::forward<Executor_>(exec), params_);
        }

        /// Create a new par_simd_policy_shim from the given
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
        /// \returns The new par_simd_policy_shim
        ///
        template <typename... Parameters_,
            typename ParametersType = typename parallel::execution::
                executor_parameters_join<Parameters_...>::type>
        typename parallel::execution::rebind_executor<par_simd_policy_shim,
            executor_type, ParametersType>::type
        with(Parameters_&&... params) const
        {
            typedef typename parallel::execution::rebind_executor<
                par_simd_policy_shim, executor_type, ParametersType>::type
                rebound_type;
            return rebound_type(exec_,
                parallel::execution::join_executor_parameters(
                    std::forward<Parameters_>(params)...));
        }

        /// Return the associated executor object.
        Executor& executor()
        {
            return exec_;
        }
        /// Return the associated executor object.
        constexpr Executor const& executor() const
        {
            return exec_;
        }

        /// Return the associated executor parameters object.
        Parameters& parameters()
        {
            return params_;
        }
        /// Return the associated executor parameters object.
        constexpr Parameters const& parameters() const
        {
            return params_;
        }

        /// \cond NOINTERNAL
        constexpr par_simd_policy_shim() = default;

        template <typename Executor_, typename Parameters_>
        constexpr par_simd_policy_shim(Executor_&& exec, Parameters_&& params)
          : exec_(std::forward<Executor_>(exec))
          , params_(std::forward<Parameters_>(params))
        {
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int /* version */)
        {
            // clang-format off
            ar & exec_ & params_;
            // clang-format on
        }

    private:
        Executor exec_;
        Parameters params_;
        /// \endcond
    };

    /// The class par_simd_task_policy_shim is an
    /// execution policy type used as a unique type to disambiguate parallel
    /// algorithm overloading based on combining a underlying
    /// \a par_simd_task_policy and an executor and indicate that
    /// a parallel algorithm's execution may be parallelized and vectorized.
    template <typename Executor, typename Parameters>
    struct par_simd_task_policy_shim : par_simd_task_policy
    {
        /// The type of the executor associated with this execution policy
        typedef Executor executor_type;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        typedef Parameters executor_parameters_type;

        /// The category of the execution agents created by this execution
        /// policy.
        typedef typename hpx::traits::executor_execution_category<
            executor_type>::type execution_category;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            typedef par_simd_task_policy_shim<Executor_, Parameters_> type;
        };

        /// Create a new par_simd_task_policy_shim from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequenced_task_policy
        ///
        constexpr par_simd_task_policy_shim operator()(task_policy_tag) const
        {
            return *this;
        }

        /// Create a new non task policy from itself
        ///
        /// \returns The non task par_simd policy
        ///
        inline decltype(auto) operator()(non_task_policy_tag /*tag*/) const;

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
        typename parallel::execution::rebind_executor<par_simd_task_policy_shim,
            Executor_, executor_parameters_type>::type
        on(Executor_&& exec) const
        {
            static_assert(hpx::traits::is_executor_any<Executor_>::value,
                "hpx::traits::is_executor_any<Executor_>::value");

            typedef typename parallel::execution::rebind_executor<
                par_simd_task_policy_shim, Executor_,
                executor_parameters_type>::type rebound_type;
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
        template <typename... Parameters_,
            typename ParametersType = typename parallel::execution::
                executor_parameters_join<Parameters_...>::type>
        typename parallel::execution::rebind_executor<par_simd_task_policy_shim,
            executor_type, ParametersType>::type
        with(Parameters_&&... params) const
        {
            typedef typename parallel::execution::rebind_executor<
                par_simd_task_policy_shim, executor_type, ParametersType>::type
                rebound_type;
            return rebound_type(exec_,
                parallel::execution::join_executor_parameters(
                    std::forward<Parameters_>(params)...));
        }

        /// Return the associated executor object.
        Executor& executor()
        {
            return exec_;
        }
        /// Return the associated executor object.
        constexpr Executor const& executor() const
        {
            return exec_;
        }

        /// Return the associated executor parameters object.
        Parameters& parameters()
        {
            return params_;
        }
        /// Return the associated executor parameters object.
        constexpr Parameters const& parameters() const
        {
            return params_;
        }

        /// \cond NOINTERNAL
        constexpr par_simd_task_policy_shim() = default;

        template <typename Executor_, typename Parameters_>
        constexpr par_simd_task_policy_shim(
            Executor_&& exec, Parameters_&& params)
          : exec_(std::forward<Executor_>(exec))
          , params_(std::forward<Parameters_>(params))
        {
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int /* version */)
        {
            // clang-format off
            ar & exec_ & params_;
            // clang-format on
        }

    private:
        Executor exec_;
        Parameters params_;
        /// \endcond
    };

    decltype(auto) simd_task_policy::operator()(
        non_task_policy_tag /*tag*/) const
    {
        return simd.on(executor()).with(parameters());
    }

    decltype(auto) par_simd_task_policy::operator()(
        non_task_policy_tag /*tag*/) const
    {
        return par_simd.on(executor()).with(parameters());
    }

    template <typename Executor, typename Parameters>
    decltype(auto) simd_task_policy_shim<Executor, Parameters>::operator()(
        non_task_policy_tag /*tag*/) const
    {
        return simd_policy_shim<Executor, Parameters>{}
            .on(executor())
            .with(parameters());
    }

    template <typename Executor, typename Parameters>
    decltype(auto) par_simd_task_policy_shim<Executor, Parameters>::operator()(
        non_task_policy_tag /*tag*/) const
    {
        return par_simd_policy_shim<Executor, Parameters>{}
            .on(executor())
            .with(parameters());
    }
}}}    // namespace hpx::execution::v1

namespace hpx { namespace detail {
    ///////////////////////////////////////////////////////////////////////////
    // extensions

    /// \cond NOINTERNAL
    template <>
    struct is_execution_policy<hpx::execution::simd_policy> : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_execution_policy<
        hpx::execution::simd_policy_shim<Executor, Parameters>> : std::true_type
    {
    };

    template <>
    struct is_execution_policy<hpx::execution::simd_task_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_execution_policy<
        hpx::execution::simd_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <>
    struct is_execution_policy<hpx::execution::par_simd_policy> : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_execution_policy<
        hpx::execution::par_simd_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <>
    struct is_execution_policy<hpx::execution::par_simd_task_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_execution_policy<
        hpx::execution::par_simd_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    template <>
    struct is_sequenced_execution_policy<hpx::execution::simd_policy>
      : std::true_type
    {
    };

    template <>
    struct is_sequenced_execution_policy<hpx::execution::simd_task_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_sequenced_execution_policy<
        hpx::execution::simd_policy_shim<Executor, Parameters>> : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_sequenced_execution_policy<
        hpx::execution::simd_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    template <>
    struct is_async_execution_policy<hpx::execution::simd_task_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_async_execution_policy<
        hpx::execution::simd_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <>
    struct is_async_execution_policy<hpx::execution::par_simd_task_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_async_execution_policy<
        hpx::execution::par_simd_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    template <>
    struct is_parallel_execution_policy<hpx::execution::par_simd_policy>
      : std::true_type
    {
    };

    template <>
    struct is_parallel_execution_policy<hpx::execution::par_simd_task_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_parallel_execution_policy<
        hpx::execution::par_simd_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_parallel_execution_policy<
        hpx::execution::par_simd_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    template <>
    struct is_vectorpack_execution_policy<hpx::execution::simd_policy>
      : std::true_type
    {
    };

    template <>
    struct is_vectorpack_execution_policy<hpx::execution::simd_task_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_vectorpack_execution_policy<
        hpx::execution::simd_policy_shim<Executor, Parameters>> : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_vectorpack_execution_policy<
        hpx::execution::simd_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <>
    struct is_vectorpack_execution_policy<hpx::execution::par_simd_policy>
      : std::true_type
    {
    };

    template <>
    struct is_vectorpack_execution_policy<hpx::execution::par_simd_task_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_vectorpack_execution_policy<
        hpx::execution::par_simd_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_vectorpack_execution_policy<
        hpx::execution::par_simd_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };
    /// \endcond
}}    // namespace hpx::detail

#endif
