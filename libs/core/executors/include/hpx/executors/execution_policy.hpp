//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2016 Marcin Copik
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/execution/execution_policy.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution/executors/rebind_executor.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/execution_base/traits/is_executor_parameters.hpp>
#include <hpx/executors/datapar/execution_policy.hpp>
#include <hpx/executors/execution_policy_fwd.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/executors/sequenced_executor.hpp>
#include <hpx/serialization/serialize.hpp>

#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution {

    ///////////////////////////////////////////////////////////////////////////
    /// Default sequential execution policy object.
    HPX_INLINE_CONSTEXPR_VARIABLE task_policy_tag task{};

    HPX_INLINE_CONSTEXPR_VARIABLE non_task_policy_tag non_task{};

    namespace detail {
        template <typename T, typename Enable = void>
        struct has_async_execution_policy : std::false_type
        {
        };

        template <typename T>
        struct has_async_execution_policy<T,
            std::void_t<decltype(std::declval<std::decay_t<T>>()(task))>>
          : std::true_type
        {
        };

        template <typename T>
        inline constexpr bool has_async_execution_policy_v =
            has_async_execution_policy<T>::value;
    }    // namespace detail

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
        using executor_type = sequenced_executor;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        using executor_parameters_type =
            parallel::execution::extract_executor_parameters<
                executor_type>::type;

        /// The category of the execution agents created by this execution
        /// policy.
        using execution_category = sequenced_execution_tag;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            using type = sequenced_task_policy_shim<Executor_, Parameters_>;
        };

        /// \cond NOINTERNAL
        constexpr sequenced_task_policy() = default;
        /// \endcond

        /// Create a new sequenced_task_policy from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequenced_task_policy
        ///
        constexpr sequenced_task_policy operator()(
            task_policy_tag /*tag*/) const
        {
            return *this;
        }

        /// Create a corresponding non task policy for this task policy
        ///
        /// \returns The non task seqeuential policy
        ///
        inline constexpr decltype(auto) operator()(
            non_task_policy_tag /*tag*/) const;

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
        constexpr decltype(auto) on(Executor&& exec) const
        {
            using executor_type = std::decay_t<Executor>;

            static_assert(hpx::traits::is_executor_any<executor_type>::value,
                "hpx::traits::is_executor_any<Executor>::value");

            return hpx::parallel::execution::create_rebound_policy(
                *this, std::forward<Executor>(exec), parameters());
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
        template <typename... Parameters>
        constexpr decltype(auto) with(Parameters&&... params) const
        {
            return hpx::parallel::execution::create_rebound_policy(*this,
                executor(),
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
        friend struct hpx::parallel::execution::create_rebound_policy_t;
        friend class hpx::serialization::access;

        template <typename Archive>
        constexpr void serialize(Archive&, const unsigned int)
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
    {
        /// The type of the executor associated with this execution policy
        using executor_type = std::decay_t<Executor>;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        using executor_parameters_type = std::decay_t<Parameters>;

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
            using type = sequenced_task_policy_shim<Executor_, Parameters_>;
        };

        /// Create a new sequenced_task_policy from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequenced_task_policy
        ///
        constexpr sequenced_task_policy_shim const& operator()(
            task_policy_tag /* tag */) const
        {
            return *this;
        }

        /// Create a corresponding non task policy for this task policy
        ///
        /// \returns The non task seqeuential shim policy
        ///
        inline constexpr decltype(auto) operator()(
            non_task_policy_tag /*tag*/) const;

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
        constexpr decltype(auto) on(Executor_&& exec) const
        {
            using executor_type = std::decay_t<Executor_>;

            static_assert(hpx::traits::is_executor_any<executor_type>::value,
                "hpx::traits::is_executor_any<Executor>::value");

            return hpx::parallel::execution::create_rebound_policy(
                *this, std::forward<Executor_>(exec), parameters());
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
        template <typename... Parameters_>
        constexpr decltype(auto) with(Parameters_&&... params) const
        {
            return hpx::parallel::execution::create_rebound_policy(*this,
                executor(),
                parallel::execution::join_executor_parameters(
                    std::forward<Parameters_>(params)...));
        }

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

        /// \cond NOINTERNAL
        template <typename Dependent = void,
            typename Enable =
                std::enable_if_t<std::is_constructible<Executor>::value &&
                        std::is_constructible<Parameters>::value,
                    Dependent>>
        constexpr sequenced_task_policy_shim()
        {
        }

        template <typename Executor_, typename Parameters_>
        constexpr sequenced_task_policy_shim(
            Executor_&& exec, Parameters_&& params)
          : exec_(std::forward<Executor_>(exec))
          , params_(std::forward<Parameters_>(params))
        {
        }

    private:
        friend struct hpx::parallel::execution::create_rebound_policy_t;
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            // clang-format off
            ar & exec_ & params_;
            // clang-format on
        }

    private:
        executor_type exec_;
        executor_parameters_type params_;
        /// \endcond
    };

    ///////////////////////////////////////////////////////////////////////////
    /// The class sequenced_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// require that a parallel algorithm's execution may not be parallelized.
    struct sequenced_policy
    {
        /// The type of the executor associated with this execution policy
        using executor_type = sequenced_executor;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        using executor_parameters_type =
            parallel::execution::extract_executor_parameters<
                executor_type>::type;

        /// The category of the execution agents created by this execution
        /// policy.
        using execution_category = sequenced_execution_tag;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            using type = sequenced_policy_shim<Executor_, Parameters_>;
        };

        /// \cond NOINTERNAL
        constexpr sequenced_policy() = default;
        /// \endcond

        /// Create a new sequenced_task_policy.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequenced_task_policy
        ///
        constexpr sequenced_task_policy operator()(
            task_policy_tag /*tag*/) const
        {
            return sequenced_task_policy();
        }

        /// Create a new sequenced_policy from itself.
        ///
        /// \returns The new sequenced_policy
        ///
        constexpr decltype(auto) operator()(non_task_policy_tag /*tag*/) const
        {
            return *this;
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
        constexpr decltype(auto) on(Executor&& exec) const
        {
            using executor_type = std::decay_t<Executor>;

            static_assert(hpx::traits::is_executor_any<executor_type>::value,
                "hpx::traits::is_executor_any<Executor>::value");

            return hpx::parallel::execution::create_rebound_policy(
                *this, std::forward<Executor>(exec), parameters());
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
        template <typename... Parameters>
        constexpr decltype(auto) with(Parameters&&... params) const
        {
            return hpx::parallel::execution::create_rebound_policy(*this,
                executor(),
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
        friend struct hpx::parallel::execution::create_rebound_policy_t;
        friend class hpx::serialization::access;

        template <typename Archive>
        constexpr void serialize(Archive&, const unsigned int)
        {
        }

    private:
        executor_type exec_;
        executor_parameters_type params_;
    };

    /// Default sequential execution policy object.
    HPX_INLINE_CONSTEXPR_VARIABLE sequenced_policy seq{};

    /// The class sequenced_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// require that a parallel algorithm's execution may not be parallelized.
    template <typename Executor, typename Parameters>
    struct sequenced_policy_shim
    {
        /// The type of the executor associated with this execution policy
        using executor_type = std::decay_t<Executor>;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        using executor_parameters_type = std::decay_t<Parameters>;

        /// The category of the execution agents created by this execution
        /// policy.
        using execution_category =
            typename hpx::traits::executor_execution_category<
                executor_type>::type;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            using type = sequenced_policy_shim<Executor_, Parameters_>;
        };

        /// Create a new sequenced_task_policy.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequenced_task_policy_shim
        ///
        constexpr sequenced_task_policy_shim<Executor, Parameters> operator()(
            task_policy_tag /* tag */) const
        {
            return sequenced_task_policy_shim<Executor, Parameters>(
                exec_, params_);
        }

        /// Create a new sequenced_policy from itself.
        ///
        /// \returns The new sequenced_policy_shim
        ///
        constexpr decltype(auto) operator()(non_task_policy_tag /*tag*/) const
        {
            return *this;
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
        constexpr decltype(auto) on(Executor_&& exec) const
        {
            using executor_type = std::decay_t<Executor_>;

            static_assert(hpx::traits::is_executor_any<executor_type>::value,
                "hpx::traits::is_executor_any<Executor>::value");

            return hpx::parallel::execution::create_rebound_policy(
                *this, std::forward<Executor_>(exec), parameters());
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
        template <typename... Parameters_>
        constexpr decltype(auto) with(Parameters_&&... params) const
        {
            return hpx::parallel::execution::create_rebound_policy(*this,
                executor(),
                parallel::execution::join_executor_parameters(
                    std::forward<Parameters_>(params)...));
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

        /// \cond NOINTERNAL
        template <typename Dependent = void,
            typename Enable =
                std::enable_if_t<std::is_constructible<Executor>::value &&
                        std::is_constructible<Parameters>::value,
                    Dependent>>
        constexpr sequenced_policy_shim()
        {
        }

        template <typename Executor_, typename Parameters_>
        constexpr sequenced_policy_shim(Executor_&& exec, Parameters_&& params)
          : exec_(std::forward<Executor_>(exec))
          , params_(std::forward<Parameters_>(params))
        {
        }

    private:
        friend struct hpx::parallel::execution::create_rebound_policy_t;
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            // clang-format off
            ar & exec_ & params_;
            // clang-format on
        }

    private:
        executor_type exec_;
        executor_parameters_type params_;
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
        using executor_type = parallel_executor;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        using executor_parameters_type =
            parallel::execution::extract_executor_parameters<
                executor_type>::type;

        /// The category of the execution agents created by this execution
        /// policy.
        using execution_category = parallel_execution_tag;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            using type = parallel_task_policy_shim<Executor_, Parameters_>;
        };

        /// \cond NOINTERNAL
        constexpr parallel_task_policy() = default;
        /// \endcond

        /// Create a new parallel_task_policy from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new parallel_task_policy
        ///
        constexpr parallel_task_policy operator()(task_policy_tag /*tag*/) const
        {
            return *this;
        }

        /// Create a new non task parallel policy from itself
        ///
        /// \param tag          [in] Specify that the corresponding
        ///                     execution policy should be used
        ///
        /// \returns The new non task parallel_policy
        ///
        inline constexpr decltype(auto) operator()(
            non_task_policy_tag /*tag*/) const;

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
        constexpr decltype(auto) on(Executor&& exec) const
        {
            using executor_type = std::decay_t<Executor>;

            static_assert(hpx::traits::is_executor_any<executor_type>::value,
                "hpx::traits::is_executor_any<Executor>::value");

            return hpx::parallel::execution::create_rebound_policy(
                *this, std::forward<Executor>(exec), parameters());
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
        template <typename... Parameters>
        constexpr decltype(auto) with(Parameters&&... params) const
        {
            return hpx::parallel::execution::create_rebound_policy(*this,
                executor(),
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
        friend struct hpx::parallel::execution::create_rebound_policy_t;
        friend class hpx::serialization::access;

        template <typename Archive>
        constexpr void serialize(Archive&, const unsigned int)
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
    struct parallel_task_policy_shim
    {
        /// The type of the executor associated with this execution policy
        using executor_type = std::decay_t<Executor>;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        using executor_parameters_type = std::decay_t<Parameters>;

        /// The category of the execution agents created by this execution
        /// policy.
        using execution_category =
            typename hpx::traits::executor_execution_category<
                executor_type>::type;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            using type = parallel_task_policy_shim<Executor_, Parameters_>;
        };

        /// Create a new parallel_task_policy_shim from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new sequenced_task_policy
        ///
        constexpr parallel_task_policy_shim operator()(
            task_policy_tag /* tag */) const
        {
            return *this;
        }

        /// Create a new non task parallel policy from itself
        ///
        /// \param tag          [in] Specify that the corresponding
        ///                     execution policy should be used
        ///
        /// \returns The new non task parallel_policy_shim
        ///
        inline constexpr decltype(auto) operator()(
            non_task_policy_tag /*tag*/) const;

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
        constexpr decltype(auto) on(Executor_&& exec) const
        {
            using executor_type = std::decay_t<Executor_>;

            static_assert(hpx::traits::is_executor_any<executor_type>::value,
                "hpx::traits::is_executor_any<Executor>::value");

            return hpx::parallel::execution::create_rebound_policy(
                *this, std::forward<Executor_>(exec), parameters());
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
        template <typename... Parameters_>
        constexpr decltype(auto) with(Parameters_&&... params) const
        {
            return hpx::parallel::execution::create_rebound_policy(*this,
                executor(),
                parallel::execution::join_executor_parameters(
                    std::forward<Parameters_>(params)...));
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

        /// \cond NOINTERNAL
        template <typename Dependent = void,
            typename Enable =
                std::enable_if_t<std::is_constructible<Executor>::value &&
                        std::is_constructible<Parameters>::value,
                    Dependent>>
        constexpr parallel_task_policy_shim()
        {
        }

        template <typename Executor_, typename Parameters_>
        constexpr parallel_task_policy_shim(
            Executor_&& exec, Parameters_&& params)
          : exec_(std::forward<Executor_>(exec))
          , params_(std::forward<Parameters_>(params))
        {
        }

    private:
        friend struct hpx::parallel::execution::create_rebound_policy_t;
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            // clang-format off
            ar & exec_ & params_;
            // clang-format on
        }

    private:
        executor_type exec_;
        executor_parameters_type params_;
        /// \endcond
    };

    ///////////////////////////////////////////////////////////////////////////
    /// The class parallel_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// indicate that a parallel algorithm's execution may be parallelized.
    struct parallel_policy
    {
        /// The type of the executor associated with this execution policy
        using executor_type = parallel_executor;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        using executor_parameters_type =
            parallel::execution::extract_executor_parameters<
                executor_type>::type;

        /// The category of the execution agents created by this execution
        /// policy.
        using execution_category = parallel_execution_tag;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            using type = parallel_policy_shim<Executor_, Parameters_>;
        };

        /// \cond NOINTERNAL
        constexpr parallel_policy() = default;
        /// \endcond

        /// Create a new parallel_policy referencing a chunk size.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new parallel_policy
        ///
        constexpr parallel_task_policy operator()(task_policy_tag /*tag*/) const
        {
            return parallel_task_policy();
        }

        /// Create a new parallel_policy from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new parallel_policy
        ///
        constexpr decltype(auto) operator()(non_task_policy_tag /*tag*/) const
        {
            return *this;
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
        constexpr decltype(auto) on(Executor&& exec) const
        {
            using executor_type = std::decay_t<Executor>;

            static_assert(hpx::traits::is_executor_any<executor_type>::value,
                "hpx::traits::is_executor_any<Executor>::value");

            return hpx::parallel::execution::create_rebound_policy(
                *this, std::forward<Executor>(exec), parameters());
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
        template <typename... Parameters>
        constexpr decltype(auto) with(Parameters&&... params) const
        {
            return hpx::parallel::execution::create_rebound_policy(*this,
                executor(),
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
        friend struct hpx::parallel::execution::create_rebound_policy_t;
        friend class hpx::serialization::access;

        template <typename Archive>
        constexpr void serialize(Archive&, const unsigned int)
        {
        }

    private:
        executor_type exec_;
        executor_parameters_type params_;
    };

    /// Default parallel execution policy object.
    static constexpr parallel_policy par{};

    /// The class parallel_policy_shim is an execution policy type
    /// used as a unique type to disambiguate parallel algorithm overloading
    /// and indicate that a parallel algorithm's execution may be parallelized.
    template <typename Executor, typename Parameters>
    struct parallel_policy_shim
    {
        /// The type of the executor associated with this execution policy
        using executor_type = std::decay_t<Executor>;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        using executor_parameters_type = std::decay_t<Parameters>;

        /// The category of the execution agents created by this execution
        /// policy.
        using execution_category =
            typename hpx::traits::executor_execution_category<
                executor_type>::type;

        /// Rebind the type of executor used by this execution policy. The
        /// execution category of Executor shall not be weaker than that of
        /// this execution policy
        template <typename Executor_, typename Parameters_>
        struct rebind
        {
            /// The type of the rebound execution policy
            using type = parallel_policy_shim<Executor_, Parameters_>;
        };

        /// Create a new parallel_policy referencing a chunk size.
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new parallel_policy
        ///
        constexpr parallel_task_policy_shim<Executor, Parameters> operator()(
            task_policy_tag /* tag */) const
        {
            return parallel_task_policy_shim<Executor, Parameters>(
                exec_, params_);
        }

        /// Create a new parallel_policy from itself
        ///
        /// \param tag          [in] Specify that the corresponding asynchronous
        ///                     execution policy should be used
        ///
        /// \returns The new parallel_policy
        ///
        constexpr decltype(auto) operator()(non_task_policy_tag /*tag*/) const
        {
            return *this;
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
        constexpr decltype(auto) on(Executor_&& exec) const
        {
            using executor_type = std::decay_t<Executor_>;

            static_assert(hpx::traits::is_executor_any<executor_type>::value,
                "hpx::traits::is_executor_any<Executor>::value");

            return hpx::parallel::execution::create_rebound_policy(
                *this, std::forward<Executor_>(exec), parameters());
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
        template <typename... Parameters_>
        constexpr decltype(auto) with(Parameters_&&... params) const
        {
            return hpx::parallel::execution::create_rebound_policy(*this,
                executor(),
                parallel::execution::join_executor_parameters(
                    std::forward<Parameters_>(params)...));
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

        /// \cond NOINTERNAL
        template <typename Dependent = void,
            typename Enable =
                std::enable_if_t<std::is_constructible<Executor>::value &&
                        std::is_constructible<Parameters>::value,
                    Dependent>>
        constexpr parallel_policy_shim()
        {
        }

        template <typename Executor_, typename Parameters_>
        constexpr parallel_policy_shim(Executor_&& exec, Parameters_&& params)
          : exec_(std::forward<Executor_>(exec))
          , params_(std::forward<Parameters_>(params))
        {
        }

    private:
        friend struct hpx::parallel::execution::create_rebound_policy_t;
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            // clang-format off
            ar & exec_ & params_;
            // clang-format on
        }

    private:
        executor_type exec_;
        executor_parameters_type params_;
        /// \endcond
    };

    ///////////////////////////////////////////////////////////////////////////
    /// The class parallel_unsequenced_policy is an execution policy type
    /// used as a unique type to disambiguate parallel algorithm overloading
    /// and indicate that a parallel algorithm's execution may be parallelized
    /// and vectorized.
    struct parallel_unsequenced_policy
    {
        /// The type of the executor associated with this execution policy
        using executor_type = parallel_executor;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        using executor_parameters_type =
            parallel::execution::extract_executor_parameters<
                executor_type>::type;

        /// The category of the execution agents created by this execution
        /// policy.
        using execution_category = parallel_execution_tag;

        /// \cond NOINTERNAL
        constexpr parallel_unsequenced_policy() = default;
        /// \endcond

        /// Create a new non task policy from itself
        ///
        /// \returns The non task parallel unsequenced policy
        ///
        constexpr decltype(auto) operator()(non_task_policy_tag /*tag*/) const
        {
            return *this;
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
        friend struct hpx::parallel::execution::create_rebound_policy_t;
        friend class hpx::serialization::access;

        template <typename Archive>
        constexpr void serialize(Archive&, const unsigned int)
        {
        }

    private:
        executor_type exec_;
        executor_parameters_type params_;
    };

    /// Default vector execution policy object.
    HPX_INLINE_CONSTEXPR_VARIABLE parallel_unsequenced_policy par_unseq{};

    ///////////////////////////////////////////////////////////////////////////
    /// The class unsequenced_policy is an execution policy type
    /// used as a unique type to disambiguate parallel algorithm overloading
    /// and indicate that a parallel algorithm's execution may be vectorized.
    struct unsequenced_policy
    {
        /// The type of the executor associated with this execution policy
        using executor_type = sequenced_executor;

        /// The type of the associated executor parameters object which is
        /// associated with this execution policy
        using executor_parameters_type =
            parallel::execution::extract_executor_parameters<
                executor_type>::type;

        /// The category of the execution agents created by this execution
        /// policy.
        using execution_category = sequenced_execution_tag;

        /// \cond NOINTERNAL
        constexpr unsequenced_policy() = default;
        /// \endcond

        /// Create a new non task policy from itself
        ///
        /// \returns The non task unsequenced policy
        ///
        constexpr decltype(auto) operator()(non_task_policy_tag /*tag*/) const
        {
            return *this;
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
        friend struct hpx::parallel::execution::create_rebound_policy_t;
        friend class hpx::serialization::access;

        template <typename Archive>
        constexpr void serialize(Archive&, const unsigned int)
        {
        }

    private:
        executor_type exec_;
        executor_parameters_type params_;
    };

    /// Default vector execution policy object.
    HPX_INLINE_CONSTEXPR_VARIABLE unsequenced_policy unseq{};

    constexpr decltype(auto) sequenced_task_policy::operator()(
        non_task_policy_tag /*tag*/) const
    {
        return seq.on(executor()).with(parameters());
    }

    constexpr decltype(auto) parallel_task_policy::operator()(
        non_task_policy_tag /*tag*/) const
    {
        return par.on(executor()).with(parameters());
    }

    template <typename Executor, typename Parameters>
    constexpr decltype(auto)
        sequenced_task_policy_shim<Executor, Parameters>::operator()(
            non_task_policy_tag /*tag*/) const
    {
        return sequenced_policy_shim<Executor, Parameters>{}
            .on(executor())
            .with(parameters());
    }

    template <typename Executor, typename Parameters>
    constexpr decltype(auto)
        parallel_task_policy_shim<Executor, Parameters>::operator()(
            non_task_policy_tag /*tag*/) const
    {
        return parallel_policy_shim<Executor, Parameters>{}
            .on(executor())
            .with(parameters());
    }

}}    // namespace hpx::execution

namespace hpx { namespace parallel { namespace execution {

    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::par is deprecated. Please use "
        "hpx::execution::par instead.")
    HPX_INLINE_CONSTEXPR_VARIABLE hpx::execution::parallel_policy par{};
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::par_unseq is deprecated. Please use "
        "hpx::execution::par_unseq instead.")
    HPX_INLINE_CONSTEXPR_VARIABLE hpx::execution::parallel_policy par_unseq{};
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::seq is deprecated. Please use "
        "hpx::execution::seq instead.")
    HPX_INLINE_CONSTEXPR_VARIABLE hpx::execution::sequenced_policy seq{};
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::task is deprecated. Please use "
        "hpx::execution::task instead.")
    HPX_INLINE_CONSTEXPR_VARIABLE hpx::execution::task_policy_tag task{};
    using parallel_executor HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::parallel_executor is deprecated. Please "
        "use "
        "hpx::execution::parallel_executor instead.") =
        hpx::execution::parallel_executor;
    using parallel_policy HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::parallel_policy is deprecated. Please "
        "use "
        "hpx::execution::parallel_policy instead.") =
        hpx::execution::parallel_policy;
    template <typename Executor, typename Parameters>
    using parallel_policy_shim HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::parallel_policy_shim is deprecated. "
        "Please "
        "use hpx::execution::parallel_policy_shim instead.") =
        hpx::execution::parallel_policy_shim<Executor, Parameters>;
    using parallel_task_policy HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::parallel_task_policy is deprecated. "
        "Please "
        "use hpx::execution::parallel_task_policy instead.") =
        hpx::execution::parallel_task_policy;
    template <typename Executor, typename Parameters>
    using parallel_task_policy_shim HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::parallel_task_policy_shim is "
        "deprecated. "
        "Please use hpx::execution::parallel_task_policy_shim instead.") =
        hpx::execution::parallel_task_policy_shim<Executor, Parameters>;
    using parallel_unsequenced_policy HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::parallel_unsequenced_policy is "
        "deprecated. "
        "Please use hpx::execution::parallel_unsequenced_policy instead.") =
        hpx::execution::parallel_unsequenced_policy;
    using sequenced_executor HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::sequenced_executor is deprecated. "
        "Please "
        "use hpx::execution::sequenced_executor instead.") =
        hpx::execution::sequenced_executor;
    using sequenced_policy HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::sequenced_policy is deprecated. Please "
        "use "
        "hpx::execution::sequenced_policy instead.") =
        hpx::execution::sequenced_policy;
    template <typename Executor, typename Parameters>
    using sequenced_policy_shim HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::sequenced_policy_shim is deprecated. "
        "Please "
        "use hpx::execution::sequenced_policy_shim instead.") =
        hpx::execution::sequenced_policy_shim<Executor, Parameters>;
    using sequenced_task_policy HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::sequenced_task_policy is deprecated. "
        "Please "
        "use hpx::execution::sequenced_task_policy instead.") =
        hpx::execution::sequenced_task_policy;
    template <typename Executor, typename Parameters>
    using sequenced_task_policy_shim HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::sequenced_task_policy_shim is "
        "deprecated. "
        "Please use hpx::execution::sequenced_task_policy_shim instead.") =
        hpx::execution::sequenced_task_policy_shim<Executor, Parameters>;
}}}    // namespace hpx::parallel::execution

namespace hpx { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    // Allow to detect execution policies which were created as a result
    // of a rebind operation. This information can be used to inhibit the
    // construction of a generic execution_policy from any of the rebound
    // policies.
    template <typename Executor, typename Parameters>
    struct is_rebound_execution_policy<
        hpx::execution::sequenced_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_rebound_execution_policy<
        hpx::execution::sequenced_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_rebound_execution_policy<
        hpx::execution::parallel_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_rebound_execution_policy<
        hpx::execution::parallel_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    template <>
    struct is_execution_policy<hpx::execution::parallel_policy> : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_execution_policy<
        hpx::execution::parallel_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <>
    struct is_execution_policy<hpx::execution::parallel_unsequenced_policy>
      : std::true_type
    {
    };

    template <>
    struct is_execution_policy<hpx::execution::unsequenced_policy>
      : std::true_type
    {
    };

    template <>
    struct is_execution_policy<hpx::execution::sequenced_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_execution_policy<
        hpx::execution::sequenced_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    // extension
    template <>
    struct is_execution_policy<hpx::execution::sequenced_task_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_execution_policy<
        hpx::execution::sequenced_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <>
    struct is_execution_policy<hpx::execution::parallel_task_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_execution_policy<
        hpx::execution::parallel_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    template <>
    struct is_parallel_execution_policy<hpx::execution::parallel_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_parallel_execution_policy<
        hpx::execution::parallel_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <>
    struct is_parallel_execution_policy<
        hpx::execution::parallel_unsequenced_policy> : std::true_type
    {
    };

    template <>
    struct is_parallel_execution_policy<hpx::execution::parallel_task_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_parallel_execution_policy<
        hpx::execution::parallel_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    template <>
    struct is_sequenced_execution_policy<hpx::execution::sequenced_task_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_sequenced_execution_policy<
        hpx::execution::sequenced_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <>
    struct is_sequenced_execution_policy<hpx::execution::sequenced_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_sequenced_execution_policy<
        hpx::execution::sequenced_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <>
    struct is_sequenced_execution_policy<hpx::execution::unsequenced_policy>
      : std::true_type
    {
    };
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    template <>
    struct is_async_execution_policy<hpx::execution::sequenced_task_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_async_execution_policy<
        hpx::execution::sequenced_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <>
    struct is_async_execution_policy<hpx::execution::parallel_task_policy>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_async_execution_policy<
        hpx::execution::parallel_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    /// \endcond
}}    // namespace hpx::detail
