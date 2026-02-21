//  Copyright (c) 2016-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/executors/datapar/execution_policy_fwd.hpp>
#include <hpx/executors/datapar/execution_policy_mappings.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/executors/sequenced_executor.hpp>
#include <hpx/modules/async_base.hpp>
#include <hpx/modules/execution.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/properties.hpp>
#include <hpx/modules/serialization.hpp>

#include <type_traits>
#include <utility>

namespace hpx::execution {

    namespace detail {

        // Extension: The class simd_task_policy_shim is an execution policy
        // type used as a unique type to disambiguate parallel algorithm
        // overloading based on combining a underlying \a sequenced_task_policy
        // and an executor and indicate that a parallel algorithm's execution
        // may not be parallelized  (has to run sequentially).
        //
        // The algorithm returns a future representing the result of the
        // corresponding algorithm when invoked with the sequenced_policy.
        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        struct simd_task_policy_shim
          : execution_policy<simd_task_policy_shim, Executor, Parameters,
                unsequenced_execution_tag>
        {
        private:
            using base_type = execution_policy<simd_task_policy_shim, Executor,
                Parameters, unsequenced_execution_tag>;

        public:
            /// \cond NOINTERNAL
            constexpr simd_task_policy_shim() = default;

            template <typename Executor_, typename Parameters_>
            constexpr simd_task_policy_shim(
                Executor_&& exec, Parameters_&& params)
              : base_type(HPX_FORWARD(Executor_, exec),
                    HPX_FORWARD(Parameters_, params))
            {
            }

            template <typename Executor_, typename Parameters_,
                typename = std::enable_if_t<
                    !std::is_same_v<
                        simd_task_policy_shim<Executor_, Parameters_>,
                        simd_task_policy_shim> &&
                    std::is_convertible_v<Executor_, Executor> &&
                    std::is_convertible_v<Parameters_, Parameters>>>
            explicit constexpr simd_task_policy_shim(
                simd_task_policy_shim<Executor_, Parameters_> const& rhs)
              : base_type(
                    simd_task_policy_shim(rhs.executor(), rhs.parameters()))
            {
            }

            template <typename Executor_, typename Parameters_,
                typename = std::enable_if_t<
                    std::is_convertible_v<Executor_, Executor> &&
                    std::is_convertible_v<Parameters_, Parameters>>>
            simd_task_policy_shim& operator=(
                simd_task_policy_shim<Executor_, Parameters_> const& rhs)
            {
                base_type::operator=(
                    simd_task_policy_shim(rhs.executor(), rhs.parameters()));
                return *this;
            }
            /// \endcond
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    /// Extension: The class simd_task_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// indicate that a parallel algorithm's execution may not be parallelized
    /// (has to run sequentially).
    ///
    /// The algorithm returns a future representing the result of the
    /// corresponding algorithm when invoked with the sequenced_policy.
    HPX_CXX_CORE_EXPORT using simd_task_policy =
        detail::simd_task_policy_shim<sequenced_executor,
            hpx::traits::executor_parameters_type_t<sequenced_executor>>;

    namespace detail {

        // The class simd_policy is an execution policy type used as a unique
        // type to disambiguate parallel algorithm overloading and require that
        // a parallel algorithm's execution may not be parallelized.
        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        struct simd_policy_shim
          : execution_policy<simd_policy_shim, Executor, Parameters,
                unsequenced_execution_tag>
        {
        private:
            using base_type = execution_policy<simd_policy_shim, Executor,
                Parameters, unsequenced_execution_tag>;

        public:
            /// \cond NOINTERNAL
            constexpr simd_policy_shim() = default;

            template <typename Executor_, typename Parameters_>
            constexpr simd_policy_shim(Executor_&& exec, Parameters_&& params)
              : base_type(HPX_FORWARD(Executor_, exec),
                    HPX_FORWARD(Parameters_, params))
            {
            }

            template <typename Executor_, typename Parameters_,
                typename = std::enable_if_t<
                    !std::is_same_v<simd_policy_shim<Executor_, Parameters_>,
                        simd_policy_shim> &&
                    std::is_convertible_v<Executor_, Executor> &&
                    std::is_convertible_v<Parameters_, Parameters>>>
            explicit constexpr simd_policy_shim(
                simd_policy_shim<Executor_, Parameters_> const& rhs)
              : base_type(simd_policy_shim(rhs.executor(), rhs.parameters()))
            {
            }

            template <typename Executor_, typename Parameters_,
                typename = std::enable_if_t<
                    std::is_convertible_v<Executor_, Executor> &&
                    std::is_convertible_v<Parameters_, Parameters>>>
            simd_policy_shim& operator=(
                simd_policy_shim<Executor_, Parameters_> const& rhs)
            {
                base_type::operator=(
                    simd_policy_shim(rhs.executor(), rhs.parameters()));
                return *this;
            }
            /// \endcond
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    /// The class simd_policy is an execution policy type used as a unique type
    /// to disambiguate parallel algorithm overloading and require that a
    /// parallel algorithm's execution may not be parallelized.
    HPX_CXX_CORE_EXPORT using simd_policy =
        detail::simd_policy_shim<sequenced_executor,
            hpx::traits::executor_parameters_type_t<sequenced_executor>>;

    /// Default sequential execution policy object.
    HPX_CXX_CORE_EXPORT inline constexpr simd_policy simd{};

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        // The class par_simd_policy is an execution policy type used as a
        // unique type to disambiguate parallel algorithm overloading and
        // indicate that a parallel algorithm's execution may be parallelized.
        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        struct par_simd_task_policy_shim
          : execution_policy<par_simd_task_policy_shim, Executor, Parameters,
                unsequenced_execution_tag>
        {
        private:
            using base_type = execution_policy<par_simd_task_policy_shim,
                Executor, Parameters, unsequenced_execution_tag>;

        public:
            /// \cond NOINTERNAL
            constexpr par_simd_task_policy_shim() = default;

            template <typename Executor_, typename Parameters_>
            constexpr par_simd_task_policy_shim(
                Executor_&& exec, Parameters_&& params)
              : base_type(HPX_FORWARD(Executor_, exec),
                    HPX_FORWARD(Parameters_, params))
            {
            }

            template <typename Executor_, typename Parameters_,
                typename = std::enable_if_t<
                    !std::is_same_v<
                        par_simd_task_policy_shim<Executor_, Parameters_>,
                        par_simd_task_policy_shim> &&
                    std::is_convertible_v<Executor_, Executor> &&
                    std::is_convertible_v<Parameters_, Parameters>>>
            explicit constexpr par_simd_task_policy_shim(
                par_simd_task_policy_shim<Executor_, Parameters_> const& rhs)
              : base_type(
                    par_simd_task_policy_shim(rhs.executor(), rhs.parameters()))
            {
            }

            template <typename Executor_, typename Parameters_,
                typename = std::enable_if_t<
                    std::is_convertible_v<Executor_, Executor> &&
                    std::is_convertible_v<Parameters_, Parameters>>>
            par_simd_task_policy_shim& operator=(
                par_simd_task_policy_shim<Executor_, Parameters_> const& rhs)
            {
                base_type::operator=(par_simd_task_policy_shim(
                    rhs.executor(), rhs.parameters()));
                return *this;
            }
            /// \endcond
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    /// Extension: The class par_simd_task_policy is an execution policy type
    /// used as a unique type to disambiguate parallel algorithm overloading and
    /// indicate that a parallel algorithm's execution may be parallelized.
    ///
    /// The algorithm returns a future representing the result of the
    /// corresponding algorithm when invoked with the parallel_policy.
    HPX_CXX_CORE_EXPORT using par_simd_task_policy =
        detail::par_simd_task_policy_shim<parallel_executor,
            hpx::traits::executor_parameters_type_t<parallel_executor>>;

    namespace detail {

        // The class par_simd_policy_shim is an execution policy type used as a
        // unique type to disambiguate parallel algorithm overloading and
        // indicate that a parallel algorithm's execution may be parallelized.
        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        struct par_simd_policy_shim
          : execution_policy<par_simd_policy_shim, Executor, Parameters,
                unsequenced_execution_tag>
        {
        private:
            using base_type = execution_policy<par_simd_policy_shim, Executor,
                Parameters, unsequenced_execution_tag>;

        public:
            /// \cond NOINTERNAL
            constexpr par_simd_policy_shim() = default;

            template <typename Executor_, typename Parameters_>
            constexpr par_simd_policy_shim(
                Executor_&& exec, Parameters_&& params)
              : base_type(HPX_FORWARD(Executor_, exec),
                    HPX_FORWARD(Parameters_, params))
            {
            }

            template <typename Executor_, typename Parameters_,
                typename = std::enable_if_t<
                    !std::is_same_v<
                        par_simd_policy_shim<Executor_, Parameters_>,
                        par_simd_policy_shim> &&
                    std::is_convertible_v<Executor_, Executor> &&
                    std::is_convertible_v<Parameters_, Parameters>>>
            explicit constexpr par_simd_policy_shim(
                par_simd_policy_shim<Executor_, Parameters_> const& rhs)
              : base_type(
                    par_simd_policy_shim(rhs.executor(), rhs.parameters()))
            {
            }

            template <typename Executor_, typename Parameters_,
                typename = std::enable_if_t<
                    std::is_convertible_v<Executor_, Executor> &&
                    std::is_convertible_v<Parameters_, Parameters>>>
            par_simd_policy_shim& operator=(
                par_simd_policy_shim<Executor_, Parameters_> const& rhs)
            {
                base_type::operator=(
                    par_simd_policy_shim(rhs.executor(), rhs.parameters()));
                return *this;
            }
            /// \endcond
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    /// Extension: The class par_simd_task_policy is an execution policy type
    /// used as a unique type to disambiguate parallel algorithm overloading and
    /// indicate that a parallel algorithm's execution may be parallelized.
    ///
    /// The algorithm returns a future representing the result of the
    /// corresponding algorithm when invoked with the parallel_policy.
    HPX_CXX_CORE_EXPORT using par_simd_policy =
        detail::par_simd_policy_shim<parallel_executor,
            hpx::traits::executor_parameters_type_t<parallel_executor>>;

    /// Default data-parallel execution policy object.
    HPX_CXX_CORE_EXPORT inline constexpr par_simd_policy par_simd{};

    namespace detail {

        /// \cond NOINTERN
        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_task_t tag,
            simd_policy_shim<Executor, Parameters> const& policy)
        {
            return simd_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_par_t tag,
            simd_policy_shim<Executor, Parameters> const& policy)
        {
            return par_simd_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_simd_t tag,
            simd_policy_shim<Executor, Parameters> const& policy)
        {
            return sequenced_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_simd_t tag,
            sequenced_policy_shim<Executor, Parameters> const& policy)
        {
            return simd_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_task_t tag,
            simd_task_policy_shim<Executor, Parameters> const& policy)
        {
            return simd_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_par_t tag,
            simd_task_policy_shim<Executor, Parameters> const& policy)
        {
            return par_simd_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_simd_t tag,
            simd_task_policy_shim<Executor, Parameters> const& policy)
        {
            return sequenced_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_simd_t tag,
            sequenced_task_policy_shim<Executor, Parameters> const& policy)
        {
            return simd_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_task_t tag,
            par_simd_policy_shim<Executor, Parameters> const& policy)
        {
            return par_simd_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_par_t tag,
            par_simd_policy_shim<Executor, Parameters> const& policy)
        {
            return simd_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_simd_t tag,
            par_simd_policy_shim<Executor, Parameters> const& policy)
        {
            return parallel_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_simd_t tag,
            parallel_policy_shim<Executor, Parameters> const& policy)
        {
            return par_simd_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        ///////////////////////////////////////////////////////////////////////
        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_task_t tag,
            par_simd_task_policy_shim<Executor, Parameters> const& policy)
        {
            return par_simd_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_par_t tag,
            par_simd_task_policy_shim<Executor, Parameters> const& policy)
        {
            return simd_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_simd_t tag,
            par_simd_task_policy_shim<Executor, Parameters> const& policy)
        {
            return parallel_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_simd_t tag,
            parallel_task_policy_shim<Executor, Parameters> const& policy)
        {
            return par_simd_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }
        /// \endcond
    }    // namespace detail
}    // namespace hpx::execution

namespace hpx::detail {

    ///////////////////////////////////////////////////////////////////////////
    // extensions

    /// \cond NOINTERNAL
    HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
    struct is_execution_policy<
        hpx::execution::detail::simd_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
    struct is_execution_policy<
        hpx::execution::detail::simd_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
    struct is_execution_policy<
        hpx::execution::detail::par_simd_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
    struct is_execution_policy<
        hpx::execution::detail::par_simd_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
    struct is_sequenced_execution_policy<
        hpx::execution::detail::simd_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
    struct is_sequenced_execution_policy<
        hpx::execution::detail::simd_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
    struct is_async_execution_policy<
        hpx::execution::detail::simd_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
    struct is_async_execution_policy<
        hpx::execution::detail::par_simd_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
    struct is_parallel_execution_policy<
        hpx::execution::detail::par_simd_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
    struct is_parallel_execution_policy<
        hpx::execution::detail::par_simd_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
    struct is_vectorpack_execution_policy<
        hpx::execution::detail::simd_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
    struct is_vectorpack_execution_policy<
        hpx::execution::detail::simd_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
    struct is_vectorpack_execution_policy<
        hpx::execution::detail::par_simd_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    HPX_CXX_CORE_EXPORT template <typename Executor, typename Parameters>
    struct is_vectorpack_execution_policy<
        hpx::execution::detail::par_simd_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };
    /// \endcond
}    // namespace hpx::detail

#endif
