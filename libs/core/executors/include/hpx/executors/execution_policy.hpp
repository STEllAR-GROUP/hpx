//  Copyright (c) 2007-2023 Hartmut Kaiser
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
#include <hpx/executors/execution_policy_fwd.hpp>
#include <hpx/executors/execution_policy_mappings.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/executors/sequenced_executor.hpp>
#include <hpx/modules/properties.hpp>
#include <hpx/serialization/serialize.hpp>

#include <memory>
#include <type_traits>
#include <utility>

namespace hpx::execution {

    namespace detail {

        // forward declare only
        template <template <class, class> typename Derived, typename Executor,
            typename Parameters = void, typename Category = void>
        struct execution_policy;
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    struct task_policy_tag final : hpx::execution::experimental::to_task_t
    {
    private:
        // we don't want to allow using 'task' as a CPO from user code
        using hpx::execution::experimental::to_task_t::operator();

        template <template <class, class> typename Derived, typename Executor,
            typename Parameters, typename Category>
        friend struct detail::execution_policy;
    };

    inline constexpr task_policy_tag task{};

    struct non_task_policy_tag final
      : hpx::execution::experimental::to_non_task_t
    {
    private:
        // we don't want to allow using 'non_task' as a CPO from user code
        using hpx::execution::experimental::to_non_task_t::operator();

        template <template <class, class> typename Derived, typename Executor,
            typename Parameters, typename Category>
        friend struct detail::execution_policy;
    };

    inline constexpr non_task_policy_tag non_task{};

    namespace experimental {

        template <>
        struct is_execution_policy_mapping<task_policy_tag> : std::true_type
        {
        };

        template <>
        struct is_execution_policy_mapping<non_task_policy_tag> : std::true_type
        {
        };
    }    // namespace experimental

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

        ////////////////////////////////////////////////////////////////////////
        // Base execution policy
        template <template <class, class> typename Derived, typename Executor,
            typename Parameters, typename Category>
        struct execution_policy
        {
        private:
            using decayed_executor_type = std::decay_t<Executor>;
            using decayed_parameters_type = std::decay_t<Parameters>;
            using derived_type = Derived<Executor, Parameters>;

            constexpr derived_type& derived() noexcept
            {
                return static_cast<derived_type&>(*this);
            }
            constexpr derived_type const& derived() const noexcept
            {
                return static_cast<derived_type const&>(*this);
            }

        public:
            // The type of the executor associated with this execution policy
            using executor_type = decayed_executor_type;

            // The type of the associated executor parameters object which is
            // associated with this execution policy
            using executor_parameters_type =
                std::conditional_t<std::is_void_v<decayed_parameters_type>,
                    hpx::traits::executor_parameters_type_t<executor_type>,
                    decayed_parameters_type>;

            // The category of the execution agents created by this execution
            // policy.
            using execution_category =
                std::conditional_t<std::is_void_v<Category>,
                    hpx::traits::executor_execution_category_t<executor_type>,
                    Category>;

            // Rebind the type of executor used by this execution policy. The
            // execution category of Executor shall not be weaker than that of
            // this execution policy
            template <typename Executor_, typename Parameters_>
            struct rebind
            {
                using type = Derived<Executor_, Parameters_>;
            };

            constexpr execution_policy() = default;

            template <typename Executor_, typename Parameters_>
            constexpr execution_policy(Executor_&& exec, Parameters_&& params)
              : exec_(HPX_FORWARD(Executor_, exec))
              , params_(HPX_FORWARD(Parameters_, params))
            {
            }

            // Create a new execution policy using the given tag
            //
            // \param tag [in] Specify the type of the execution policy to
            //                 return
            //
            // \returns The new execution policy
            //
            template <typename Tag,
                typename Enable = std::enable_if_t<hpx::execution::
                        experimental::is_execution_policy_mapping_v<Tag>>>
            inline constexpr decltype(auto) operator()(Tag tag) const;

            // Create a new derived execution policy from the given executor
            //
            // \tparam Executor  The type of the executor to associate with this
            //                   execution policy.
            //
            // \param exec       [in] The executor to use for the execution of
            //                   the parallel algorithm the returned execution
            //                   policy is used with.
            //
            // \note Requires: is_executor_v<Executor> is true
            //
            // \returns The new execution policy
            //
            template <typename Executor_>
            constexpr decltype(auto) on(Executor_&& exec) const
            {
                static_assert(
                    hpx::traits::is_executor_any_v<std::decay_t<Executor_>>,
                    "hpx::traits::is_executor_any_v<Executor>");

                return hpx::parallel::execution::create_rebound_policy(
                    derived(), HPX_FORWARD(Executor_, exec), parameters());
            }

            // Create a new execution policy from the given execution parameters
            //
            // \tparam Parameters  The type of the executor parameters to
            //                     associate with this execution policy.
            //
            // \param params       [in] The executor parameters to use for the
            //                     execution of the parallel algorithm the
            //                     returned execution policy is used with.
            //
            // \note Requires: all parameters are executor_parameters, different
            //                 parameter types can't be duplicated
            //
            // \returns The new execution policy
            //
            template <typename... Parameters_>
            constexpr decltype(auto) with(Parameters_&&... params) const
            {
                return hpx::parallel::execution::create_rebound_policy(
                    derived(), executor(),
                    parallel::execution::join_executor_parameters(
                        HPX_FORWARD(Parameters_, params)...));
            }

        public:
            // Return the associated executor object.
            executor_type& executor() noexcept
            {
                return exec_;
            }

            // Return the associated executor object.
            constexpr executor_type const& executor() const noexcept
            {
                return exec_;
            }

            // Return the associated executor parameters object.
            executor_parameters_type& parameters() noexcept
            {
                return params_;
            }

            // Return the associated executor parameters object.
            constexpr executor_parameters_type const& parameters()
                const noexcept
            {
                return params_;
            }

        private:
            friend struct hpx::parallel::execution::create_rebound_policy_t;
            friend class hpx::serialization::access;

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const)
            {
                // clang-format off
                ar & exec_ & params_;
                // clang-format on
            }

        private:
            executor_type exec_;
            executor_parameters_type params_;
        };

        ///////////////////////////////////////////////////////////////////////
        // Extension: The class sequenced_task_policy_shim is an execution
        // policy type used as a unique type to disambiguate parallel algorithm
        // overloading based on combining a underlying \a sequenced_task_policy
        // and an executor and indicate that a parallel algorithm's execution
        // may not be parallelized  (has to run sequentially).
        //
        // The algorithm returns a future representing the result of the
        // corresponding algorithm when invoked with the sequenced_policy.
        template <typename Executor, typename Parameters>
        struct sequenced_task_policy_shim
          : execution_policy<sequenced_task_policy_shim, Executor, Parameters>
        {
        private:
            using base_type = execution_policy<sequenced_task_policy_shim,
                Executor, Parameters>;

        public:
            /// \cond NOINTERNAL
            constexpr sequenced_task_policy_shim() = default;

            template <typename Executor_, typename Parameters_>
            constexpr sequenced_task_policy_shim(
                Executor_&& exec, Parameters_&& params)
              : base_type(HPX_FORWARD(Executor_, exec),
                    HPX_FORWARD(Parameters_, params))
            {
            }
            /// \endcond
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    /// Extension: The class sequenced_task_policy is an execution policy type
    /// used as a unique type to disambiguate parallel algorithm overloading and
    /// indicate that a parallel algorithm's execution may not be parallelized
    /// (has to run sequentially).
    ///
    /// The algorithm returns a future representing the result of the
    /// corresponding algorithm when invoked with the sequenced_policy.
    using sequenced_task_policy =
        detail::sequenced_task_policy_shim<sequenced_executor>;

    namespace detail {

        // The class sequenced_policy is an execution policy type used as a
        // unique type to disambiguate parallel algorithm overloading and
        // require that a parallel algorithm's execution may not be
        // parallelized.
        template <typename Executor, typename Parameters>
        struct sequenced_policy_shim
          : execution_policy<sequenced_policy_shim, Executor, Parameters>
        {
        private:
            using base_type =
                execution_policy<sequenced_policy_shim, Executor, Parameters>;

        public:
            /// \cond NOINTERNAL
            constexpr sequenced_policy_shim() = default;

            template <typename Executor_, typename Parameters_>
            constexpr sequenced_policy_shim(
                Executor_&& exec, Parameters_&& params)
              : base_type(HPX_FORWARD(Executor_, exec),
                    HPX_FORWARD(Parameters_, params))
            {
            }
            /// \endcond
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    /// The class sequenced_policy is an execution policy type used as a unique
    /// type to disambiguate parallel algorithm overloading and require that a
    /// parallel algorithm's execution may not be parallelized.
    using sequenced_policy = detail::sequenced_policy_shim<sequenced_executor>;

    /// Default sequential execution policy object.
    inline constexpr sequenced_policy seq{};

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        // Extension: The class parallel_task_policy_shim is an execution policy
        // type used as a unique type to disambiguate parallel algorithm
        // overloading based on combining a underlying \a parallel_task_policy
        // and an executor and indicate that a parallel algorithm's execution
        // may be parallelized.
        template <typename Executor, typename Parameters>
        struct parallel_task_policy_shim
          : execution_policy<parallel_task_policy_shim, Executor, Parameters>
        {
        private:
            using base_type = execution_policy<parallel_task_policy_shim,
                Executor, Parameters>;

        public:
            /// \cond NOINTERNAL
            constexpr parallel_task_policy_shim() = default;

            template <typename Executor_, typename Parameters_>
            constexpr parallel_task_policy_shim(
                Executor_&& exec, Parameters_&& params)
              : base_type(HPX_FORWARD(Executor_, exec),
                    HPX_FORWARD(Parameters_, params))
            {
            }
            /// \endcond
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    /// Extension: The class parallel_task_policy is an execution policy type
    /// used as a unique type to disambiguate parallel algorithm overloading and
    /// indicate that a parallel algorithm's execution may be parallelized.
    ///
    /// The algorithm returns a future representing the result of the
    /// corresponding algorithm when invoked with the parallel_policy.
    using parallel_task_policy =
        detail::parallel_task_policy_shim<parallel_executor>;

    namespace detail {

        // The class parallel_policy_shim is an execution policy type used as a
        // unique type to disambiguate parallel algorithm overloading and
        // indicate that a parallel algorithm's execution may be parallelized.
        template <typename Executor, typename Parameters>
        struct parallel_policy_shim
          : execution_policy<parallel_policy_shim, Executor, Parameters>
        {
        private:
            using base_type =
                execution_policy<parallel_policy_shim, Executor, Parameters>;

        public:
            /// \cond NOINTERNAL
            constexpr parallel_policy_shim() = default;

            template <typename Executor_, typename Parameters_>
            constexpr parallel_policy_shim(
                Executor_&& exec, Parameters_&& params)
              : base_type(HPX_FORWARD(Executor_, exec),
                    HPX_FORWARD(Parameters_, params))
            {
            }
            /// \endcond
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    /// The class parallel_policy is an execution policy type used as a unique
    /// type to disambiguate parallel algorithm overloading and indicate that a
    /// parallel algorithm's execution may be parallelized.
    using parallel_policy = detail::parallel_policy_shim<parallel_executor>;

    /// Default parallel execution policy object.
    inline constexpr parallel_policy par{};

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        // The class parallel_unsequenced_task_policy_shim is an execution
        // policy type used as a unique type to disambiguate parallel algorithm
        // overloading and indicate that a parallel algorithm's execution may be
        // parallelized and vectorized.
        template <typename Executor, typename Parameters>
        struct parallel_unsequenced_task_policy_shim
          : execution_policy<parallel_unsequenced_task_policy_shim, Executor,
                Parameters>
        {
        private:
            using base_type =
                execution_policy<parallel_unsequenced_task_policy_shim,
                    Executor, Parameters>;

        public:
            /// \cond NOINTERNAL
            constexpr parallel_unsequenced_task_policy_shim() = default;

            template <typename Executor_, typename Parameters_>
            constexpr parallel_unsequenced_task_policy_shim(
                Executor_&& exec, Parameters_&& params)
              : base_type(HPX_FORWARD(Executor_, exec),
                    HPX_FORWARD(Parameters_, params))
            {
            }
            /// \endcond
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    /// The class parallel_unsequenced_task_policy is an execution policy type
    /// used as a unique type to disambiguate parallel algorithm overloading
    /// and indicate that a parallel algorithm's execution may be parallelized
    /// and vectorized.
    using parallel_unsequenced_task_policy =
        detail::parallel_unsequenced_task_policy_shim<parallel_executor>;

    namespace detail {

        // The class parallel_unsequenced_policy_shim is an execution policy type
        // used as a unique type to disambiguate parallel algorithm overloading
        // and indicate that a parallel algorithm's execution may be parallelized.
        template <typename Executor, typename Parameters>
        struct parallel_unsequenced_policy_shim
          : execution_policy<parallel_unsequenced_policy_shim, Executor,
                Parameters>
        {
        private:
            using base_type = execution_policy<parallel_unsequenced_policy_shim,
                Executor, Parameters>;

        public:
            /// \cond NOINTERNAL
            constexpr parallel_unsequenced_policy_shim() = default;

            template <typename Executor_, typename Parameters_>
            constexpr parallel_unsequenced_policy_shim(
                Executor_&& exec, Parameters_&& params)
              : base_type(HPX_FORWARD(Executor_, exec),
                    HPX_FORWARD(Parameters_, params))
            {
            }
            /// \endcond
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    /// The class parallel_unsequenced_policy is an execution policy type used
    /// as a unique type to disambiguate parallel algorithm overloading and
    /// indicate that a parallel algorithm's execution may be parallelized and
    /// vectorized.
    using parallel_unsequenced_policy =
        detail::parallel_unsequenced_policy_shim<parallel_executor>;

    /// Default vector execution policy object.
    inline constexpr parallel_unsequenced_policy par_unseq{};

    namespace detail {

        // Extension: The class unsequenced_task_policy_shim is an execution
        // policy type used as a unique type to disambiguate parallel algorithm
        // overloading based on combining a underlying \a
        // unsequenced_task_policy and an executor and indicate that a parallel
        // algorithm's execution may be vectorized.
        //
        // The algorithm returns a future representing the result of the
        // corresponding algorithm when invoked with the unsequenced_policy.
        template <typename Executor, typename Parameters>
        struct unsequenced_task_policy_shim
          : execution_policy<unsequenced_task_policy_shim, Executor, Parameters>
        {
        private:
            using base_type = execution_policy<unsequenced_task_policy_shim,
                Executor, Parameters>;

        public:
            /// \cond NOINTERNAL
            constexpr unsequenced_task_policy_shim() = default;

            template <typename Executor_, typename Parameters_>
            constexpr unsequenced_task_policy_shim(
                Executor_&& exec, Parameters_&& params)
              : base_type(HPX_FORWARD(Executor_, exec),
                    HPX_FORWARD(Parameters_, params))
            {
            }
            /// \endcond
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    /// The class unsequenced_task_policy is an execution policy type used as a
    /// unique type to disambiguate parallel algorithm overloading and indicate
    /// that a parallel algorithm's execution may be vectorized.
    using unsequenced_task_policy =
        detail::unsequenced_task_policy_shim<sequenced_executor>;

    namespace detail {

        // The class unsequenced_policy is an execution policy type used as a
        // unique type to disambiguate parallel algorithm overloading and
        // require that a parallel algorithm's execution may be vectorized.
        template <typename Executor, typename Parameters>
        struct unsequenced_policy_shim
          : execution_policy<unsequenced_policy_shim, Executor, Parameters>
        {
        private:
            using base_type =
                execution_policy<unsequenced_policy_shim, Executor, Parameters>;

        public:
            /// \cond NOINTERNAL
            constexpr unsequenced_policy_shim() = default;

            template <typename Executor_, typename Parameters_>
            constexpr unsequenced_policy_shim(
                Executor_&& exec, Parameters_&& params)
              : base_type(HPX_FORWARD(Executor_, exec),
                    HPX_FORWARD(Parameters_, params))
            {
            }
            /// \endcond
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    /// The class unsequenced_policy is an execution policy type used as a
    /// unique type to disambiguate parallel algorithm overloading and indicate
    /// that a parallel algorithm's execution may be vectorized.
    using unsequenced_policy =
        detail::unsequenced_policy_shim<sequenced_executor>;

    /// Default vector execution policy object.
    inline constexpr unsequenced_policy unseq{};

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_task_t tag,
            sequenced_policy_shim<Executor, Parameters> const& policy)
        {
            return sequenced_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_par_t tag,
            sequenced_policy_shim<Executor, Parameters> const& policy)
        {
            return parallel_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_unseq_t tag,
            sequenced_policy_shim<Executor, Parameters> const& policy)
        {
            return unsequenced_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_task_t tag,
            sequenced_task_policy_shim<Executor, Parameters> const& policy)
        {
            return sequenced_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_par_t tag,
            sequenced_task_policy_shim<Executor, Parameters> const& policy)
        {
            return parallel_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_unseq_t tag,
            sequenced_task_policy_shim<Executor, Parameters> const& policy)
        {
            return unsequenced_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_task_t tag,
            parallel_task_policy_shim<Executor, Parameters> const& policy)
        {
            return parallel_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_par_t tag,
            parallel_task_policy_shim<Executor, Parameters> const& policy)
        {
            return sequenced_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_unseq_t tag,
            parallel_task_policy_shim<Executor, Parameters> const& policy)
        {
            return parallel_unsequenced_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_task_t tag,
            parallel_policy_shim<Executor, Parameters> const& policy)
        {
            return parallel_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_par_t tag,
            parallel_policy_shim<Executor, Parameters> const& policy)
        {
            return sequenced_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_unseq_t tag,
            parallel_policy_shim<Executor, Parameters> const& policy)
        {
            return parallel_unsequenced_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_task_t tag,
            parallel_unsequenced_task_policy_shim<Executor, Parameters> const&
                policy)
        {
            return parallel_unsequenced_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_par_t tag,
            parallel_unsequenced_task_policy_shim<Executor, Parameters> const&
                policy)
        {
            return unsequenced_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_unseq_t tag,
            parallel_unsequenced_task_policy_shim<Executor, Parameters> const&
                policy)
        {
            return parallel_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_task_t tag,
            parallel_unsequenced_policy_shim<Executor, Parameters> const&
                policy)
        {
            return parallel_unsequenced_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_par_t tag,
            parallel_unsequenced_policy_shim<Executor, Parameters> const&
                policy)
        {
            return unsequenced_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_unseq_t tag,
            parallel_unsequenced_policy_shim<Executor, Parameters> const&
                policy)
        {
            return parallel_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_task_t tag,
            unsequenced_task_policy_shim<Executor, Parameters> const& policy)
        {
            return unsequenced_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_par_t tag,
            unsequenced_task_policy_shim<Executor, Parameters> const& policy)
        {
            return parallel_unsequenced_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_unseq_t tag,
            unsequenced_task_policy_shim<Executor, Parameters> const& policy)
        {
            return sequenced_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_task_t tag,
            unsequenced_policy_shim<Executor, Parameters> const& policy)
        {
            return unsequenced_task_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_par_t tag,
            unsequenced_policy_shim<Executor, Parameters> const& policy)
        {
            return parallel_unsequenced_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        template <typename Executor, typename Parameters>
        constexpr decltype(auto) tag_invoke(
            hpx::execution::experimental::to_non_unseq_t tag,
            unsequenced_policy_shim<Executor, Parameters> const& policy)
        {
            return sequenced_policy()
                .on(hpx::experimental::prefer(tag, policy.executor()))
                .with(policy.parameters());
        }

        ///////////////////////////////////////////////////////////////////////
        template <template <class, class> typename Derived, typename Executor,
            typename Parameters, typename Category>
        template <typename Tag, typename Enable>
        constexpr decltype(auto)
        execution_policy<Derived, Executor, Parameters, Category>::operator()(
            Tag tag) const
        {
            return tag(derived());
        }
    }    // namespace detail
}    // namespace hpx::execution

namespace hpx::detail {

    ///////////////////////////////////////////////////////////////////////////
    // Allow to detect execution policies which were created as a result of a
    // rebind operation. This information can be used to inhibit the
    // construction of a generic execution_policy from any of the rebound
    // policies.
    template <typename Executor, typename Parameters>
    struct is_rebound_execution_policy<
        hpx::execution::detail::sequenced_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_rebound_execution_policy<hpx::execution::detail::
            sequenced_task_policy_shim<Executor, Parameters>> : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_rebound_execution_policy<
        hpx::execution::detail::parallel_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_rebound_execution_policy<
        hpx::execution::detail::parallel_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_rebound_execution_policy<
        hpx::execution::detail::unsequenced_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_rebound_execution_policy<hpx::execution::detail::
            unsequenced_task_policy_shim<Executor, Parameters>> : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_rebound_execution_policy<hpx::execution::detail::
            parallel_unsequenced_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_rebound_execution_policy<hpx::execution::detail::
            parallel_unsequenced_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    ////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    template <typename Executor, typename Parameters>
    struct is_execution_policy<
        hpx::execution::detail::parallel_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_execution_policy<hpx::execution::detail::
            parallel_unsequenced_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_execution_policy<
        hpx::execution::detail::unsequenced_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_execution_policy<
        hpx::execution::detail::sequenced_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    // extension
    template <typename Executor, typename Parameters>
    struct is_execution_policy<hpx::execution::detail::
            sequenced_task_policy_shim<Executor, Parameters>> : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_execution_policy<
        hpx::execution::detail::parallel_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_execution_policy<hpx::execution::detail::
            unsequenced_task_policy_shim<Executor, Parameters>> : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_execution_policy<hpx::execution::detail::
            parallel_unsequenced_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    template <typename Executor, typename Parameters>
    struct is_parallel_execution_policy<
        hpx::execution::detail::parallel_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_parallel_execution_policy<hpx::execution::detail::
            parallel_unsequenced_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_parallel_execution_policy<
        hpx::execution::detail::parallel_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_parallel_execution_policy<hpx::execution::detail::
            parallel_unsequenced_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    template <typename Executor, typename Parameters>
    struct is_sequenced_execution_policy<hpx::execution::detail::
            sequenced_task_policy_shim<Executor, Parameters>> : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_sequenced_execution_policy<
        hpx::execution::detail::sequenced_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_sequenced_execution_policy<
        hpx::execution::detail::unsequenced_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_sequenced_execution_policy<hpx::execution::detail::
            unsequenced_task_policy_shim<Executor, Parameters>> : std::true_type
    {
    };
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    template <typename Executor, typename Parameters>
    struct is_async_execution_policy<hpx::execution::detail::
            sequenced_task_policy_shim<Executor, Parameters>> : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_async_execution_policy<
        hpx::execution::detail::parallel_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_async_execution_policy<hpx::execution::detail::
            unsequenced_task_policy_shim<Executor, Parameters>> : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_async_execution_policy<hpx::execution::detail::
            parallel_unsequenced_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    template <typename Executor, typename Parameters>
    struct is_unsequenced_execution_policy<
        hpx::execution::detail::unsequenced_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_unsequenced_execution_policy<hpx::execution::detail::
            parallel_unsequenced_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_unsequenced_execution_policy<hpx::execution::detail::
            unsequenced_task_policy_shim<Executor, Parameters>> : std::true_type
    {
    };

    template <typename Executor, typename Parameters>
    struct is_unsequenced_execution_policy<hpx::execution::detail::
            parallel_unsequenced_task_policy_shim<Executor, Parameters>>
      : std::true_type
    {
    };
    /// \endcond
}    // namespace hpx::detail
