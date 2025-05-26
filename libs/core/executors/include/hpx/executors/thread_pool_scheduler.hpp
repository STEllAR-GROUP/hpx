//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2022-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/algorithms/then.hpp>
#include <hpx/execution/detail/post_policy_dispatch.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution/queries/get_scheduler.hpp>
#include <hpx/execution_base/completion_scheduler.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/modules/topology.hpp>
#include <hpx/synchronization/stop_token.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/threading_base/detail/get_default_pool.hpp>
#include <hpx/threading_base/register_thread.hpp>
#include <hpx/timing/steady_clock.hpp>
#ifdef HPX_HAVE_STDEXEC
#include <hpx/execution_base/stdexec_forward.hpp>
#endif

#include <cstddef>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    namespace detail {

        template <typename Policy>
        struct get_default_scheduler_policy
        {
            static constexpr Policy call() noexcept
            {
                return Policy{};
            }
        };

        template <>
        struct get_default_scheduler_policy<hpx::launch>
        {
            static constexpr hpx::launch::async_policy call() noexcept
            {
                return hpx::launch::async_policy{};
            }
        };
    }    // namespace detail

    template <typename Policy>
    struct thread_pool_policy_scheduler
    {
        // Associate the parallel_execution_tag tag type as a default with this
        // scheduler, except if the given launch policy is synch.
        using execution_category =
            std::conditional_t<std::is_same_v<Policy, launch::sync_policy>,
                sequenced_execution_tag, parallel_execution_tag>;

        constexpr explicit thread_pool_policy_scheduler(
            Policy l = experimental::detail::get_default_scheduler_policy<
                Policy>::call())
          : policy_(l)
        {
        }

        explicit thread_pool_policy_scheduler(
            hpx::threads::thread_pool_base* pool,
            Policy l = experimental::detail::get_default_scheduler_policy<
                Policy>::call()) noexcept
          : pool_(pool)
          , policy_(l)
        {
        }

        /// \cond NOINTERNAL
        friend constexpr bool operator==(
            thread_pool_policy_scheduler const& lhs,
            thread_pool_policy_scheduler const& rhs) noexcept
        {
            return lhs.pool_ == rhs.pool_ && lhs.policy_ == rhs.policy_;
        }

        friend constexpr bool operator!=(
            thread_pool_policy_scheduler const& lhs,
            thread_pool_policy_scheduler const& rhs) noexcept
        {
            return !(lhs == rhs);
        }

        [[nodiscard]] hpx::threads::thread_pool_base* get_thread_pool()
            const noexcept
        {
            HPX_ASSERT(pool_);
            return pool_;
        }

        // clang-format off
        template <typename Executor_,
            HPX_CONCEPT_REQUIRES_(
                std::is_convertible_v<Executor_, thread_pool_policy_scheduler>
            )>
        // clang-format on
        friend constexpr auto tag_invoke(
            hpx::execution::experimental::with_processing_units_count_t,
            Executor_ const& scheduler, std::size_t num_cores) noexcept
        {
            auto scheduler_with_num_cores = scheduler;
            scheduler_with_num_cores.num_cores_ = num_cores;
            return scheduler_with_num_cores;
        }

        // clang-format off
        template <typename Parameters,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_parameters_v<Parameters>
            )>
        // clang-format on
        friend constexpr std::size_t tag_invoke(
            hpx::execution::experimental::processing_units_count_t,
            Parameters&&, thread_pool_policy_scheduler const& scheduler,
            hpx::chrono::steady_duration const& = hpx::chrono::null_duration,
            std::size_t = 0)
        {
            return scheduler.get_num_cores();
        }

        // clang-format off
        template <typename Executor_,
            HPX_CONCEPT_REQUIRES_(
                std::is_convertible_v<Executor_, thread_pool_policy_scheduler>
            )>
        // clang-format on
        friend constexpr auto tag_invoke(
            hpx::execution::experimental::with_first_core_t,
            Executor_ const& exec, std::size_t first_core) noexcept
        {
            auto exec_with_first_core = exec;
            exec_with_first_core.first_core_ = first_core;
            return exec_with_first_core;
        }

        friend constexpr std::size_t tag_invoke(
            hpx::execution::experimental::get_first_core_t,
            thread_pool_policy_scheduler const& exec) noexcept
        {
            return exec.get_first_core();
        }

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
        // support with_annotation property
        // clang-format off
        template <typename Executor_,
            HPX_CONCEPT_REQUIRES_(
                std::is_convertible_v<Executor_, thread_pool_policy_scheduler>
            )>
        // clang-format on
        friend constexpr auto tag_invoke(
            hpx::execution::experimental::with_annotation_t,
            Executor_ const& scheduler, char const* annotation)
        {
            auto sched_with_annotation = scheduler;
            sched_with_annotation.annotation_ = annotation;
            return sched_with_annotation;
        }

        // clang-format off
        template <typename Executor_,
            HPX_CONCEPT_REQUIRES_(
                std::is_convertible_v<Executor_, thread_pool_policy_scheduler>
            )>
        // clang-format on
        friend auto tag_invoke(hpx::execution::experimental::with_annotation_t,
            Executor_ const& scheduler, std::string annotation)
        {
            auto sched_with_annotation = scheduler;
            sched_with_annotation.annotation_ =
                hpx::detail::store_function_annotation(HPX_MOVE(annotation));
            return sched_with_annotation;
        }

        // support get_annotation property
        friend constexpr char const* tag_invoke(
            hpx::execution::experimental::get_annotation_t,
            thread_pool_policy_scheduler const& scheduler) noexcept
        {
            return scheduler.annotation_;
        }
#endif

        friend auto tag_invoke(
            hpx::execution::experimental::get_processing_units_mask_t,
            thread_pool_policy_scheduler const& exec)
        {
            auto pool = exec.pool_ ?
                exec.pool_ :
                threads::detail::get_self_or_default_pool();
            return pool->get_used_processing_units(exec.get_num_cores(), false);
        }

        friend auto tag_invoke(hpx::execution::experimental::get_cores_mask_t,
            thread_pool_policy_scheduler const& exec)
        {
            auto pool = exec.pool_ ?
                exec.pool_ :
                threads::detail::get_self_or_default_pool();
            return pool->get_used_processing_units(exec.get_num_cores(), true);
        }

        template <typename F>
        void execute(F&& f, Policy const& policy) const
        {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            hpx::threads::thread_description desc(f, annotation_);
#else
            hpx::threads::thread_description desc(f);
#endif
            auto pool =
                pool_ ? pool_ : threads::detail::get_self_or_default_pool();

            hpx::detail::post_policy_dispatch<Policy>::call(
                policy, desc, pool, HPX_FORWARD(F, f));
        }

        template <typename F>
        HPX_FORCEINLINE void execute(F&& f) const
        {
            execute(HPX_FORWARD(F, f), policy_);
        }

        template <typename Scheduler, typename Receiver>
        struct operation_state
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

            template <typename Scheduler_, typename Receiver_>
            operation_state(Scheduler_&& scheduler, Receiver_&& receiver)
              : scheduler(HPX_FORWARD(Scheduler_, scheduler))
              , receiver(HPX_FORWARD(Receiver_, receiver))
            {
            }

            operation_state(operation_state&&) = delete;
            operation_state(operation_state const&) = delete;
            operation_state& operator=(operation_state&&) = delete;
            operation_state& operator=(operation_state const&) = delete;

            ~operation_state() = default;

            friend void tag_invoke(start_t, operation_state& os) noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        os.scheduler.execute(
                            [receiver = HPX_MOVE(os.receiver)]() mutable {
                                hpx::execution::experimental::set_value(
                                    HPX_MOVE(receiver));
                            });
                    },
                    [&](std::exception_ptr ep) {
                        // FIXME: set_error is called on a moved-from object
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(os.receiver), HPX_MOVE(ep));
                    });
            }
        };

        template <typename Scheduler>
        struct sender
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;
#ifdef HPX_HAVE_STDEXEC
            using sender_concept = hpx::execution::experimental::sender_t;
#endif
            using completion_signatures =
                hpx::execution::experimental::completion_signatures<
                    hpx::execution::experimental::set_value_t(),
                    hpx::execution::experimental::set_error_t(
                        std::exception_ptr),
                    hpx::execution::experimental::set_stopped_t()>;

            template <typename Env>
            friend auto tag_invoke(
                hpx::execution::experimental::get_completion_signatures_t,
                sender const&, Env) noexcept -> completion_signatures;

            template <typename Receiver>
            friend operation_state<Scheduler, Receiver> tag_invoke(
                connect_t, sender&& s, Receiver&& receiver)
            {
                return {HPX_MOVE(s.scheduler), HPX_FORWARD(Receiver, receiver)};
            }

            template <typename Receiver>
            friend operation_state<Scheduler, Receiver> tag_invoke(
                connect_t, sender& s, Receiver&& receiver)
            {
                return {s.scheduler, HPX_FORWARD(Receiver, receiver)};
            }
#ifdef HPX_HAVE_STDEXEC
            struct env
            {
                std::decay_t<Scheduler> const& sched;
                // clang-format off
                template <typename CPO,
                    HPX_CONCEPT_REQUIRES_(
                        meta::value<meta::one_of<
                            CPO, set_value_t, set_stopped_t>>
                    )>
                // clang-format on
                friend constexpr auto tag_invoke(
                    hpx::execution::experimental::get_completion_scheduler_t<
                        CPO>,
                    env const& e) noexcept
                {
                    return e.sched;
                }
            };

            friend constexpr env tag_invoke(
                hpx::execution::experimental::get_env_t,
                sender const& s) noexcept
            {
                return {s.scheduler};
            };
#else
            // clang-format off
            template <typename CPO,
                HPX_CONCEPT_REQUIRES_(
                    meta::value<meta::one_of<
                        CPO, set_value_t, set_stopped_t>>
                )>
            // clang-format on
            friend constexpr auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<CPO>,
                sender const& s)
            {
                return s.scheduler;
            }
#endif
        };

        friend constexpr hpx::execution::experimental::
            forward_progress_guarantee
            tag_invoke(
                hpx::execution::experimental::get_forward_progress_guarantee_t,
                thread_pool_policy_scheduler const& sched) noexcept
        {
            if (hpx::detail::has_async_policy(sched.policy()))
            {
                return hpx::execution::experimental::
                    forward_progress_guarantee::parallel;
            }
            else
            {
                return hpx::execution::experimental::
                    forward_progress_guarantee::concurrent;
            }
        }

        void policy(Policy policy) noexcept
        {
            policy_ = HPX_MOVE(policy);
        }

        constexpr Policy const& policy() const noexcept
        {
            return policy_;
        }
        /// \endcond

        // New additions for parallel_scheduler functionality
        /// \cond NOINTERNAL
        // Alias for sender to act as parallel_sender
        template <typename Scheduler>
        struct parallel_sender : public sender<Scheduler>
        {
#ifdef HPX_HAVE_STDEXEC
            using sender_concept = hpx::execution::experimental::sender_t;
#endif

            explicit parallel_sender(Scheduler scheduler) noexcept
              : sender<Scheduler>{scheduler}
            {
            }

#ifdef HPX_HAVE_STDEXEC
            template <typename Func>
            auto then(Func&& func) &&
            {
                return then_sender<parallel_sender, std::decay_t<Func>>{
                    std::move(*this), std::forward<Func>(func)};
            }

            template <typename Func>
            auto then(Func&& func) const&
            {
                return then_sender<parallel_sender, std::decay_t<Func>>{
                    *this, std::forward<Func>(func)};
            }
#else
            template <typename Func>
            auto then(Func&&) &&
            {
                throw std::runtime_error("stdexec not enabled");
            }

            template <typename Func>
            auto then(Func&&) const&
            {
                throw std::runtime_error("stdexec not enabled");
            }
#endif
        };

        // Wrapper for receiver to handle then() functionality
        template <typename Receiver, typename Func>
        struct wrapped_receiver
        {
            Receiver receiver_;
            Func func_;

            wrapped_receiver(Receiver&& receiver, Func&& func)
              : receiver_(std::move(receiver))
              , func_(std::move(func))
            {
            }

#ifdef HPX_HAVE_STDEXEC
            void set_value() &&
            {
                try
                {
                    func_();
                    std::move(receiver_).set_value();
                }
                catch (...)
                {
                    std::move(receiver_).set_error(std::current_exception());
                }
            }

            void set_error(std::exception_ptr ep) && noexcept
            {
                std::move(receiver_).set_error(ep);
            }

            void set_stopped() && noexcept
            {
                std::move(receiver_).set_stopped();
            }
#else
            void set_value() &&
            {
                throw std::runtime_error("stdexec not enabled");
            }

            void set_error(std::exception_ptr) && noexcept
            {
                throw std::runtime_error("stdexec not enabled");
            }

            void set_stopped() && noexcept
            {
                throw std::runtime_error("stdexec not enabled");
            }
#endif
        };

        // then_sender for chaining computations
        template <typename Sender, typename Func>
        struct then_sender
        {
#ifdef HPX_HAVE_STDEXEC
            using sender_concept = hpx::execution::experimental::sender_t;
#endif

            Sender sender_;
            Func func_;

            then_sender(Sender&& sender, Func&& func)
              : sender_(std::move(sender))
              , func_(std::move(func))
            {
            }

#ifdef HPX_HAVE_STDEXEC
            auto get_env() const noexcept
            {
                return hpx::execution::experimental::get_env(sender_);
            }

            template <typename Receiver>
            struct operation_state
            {
                using wrapped_receiver_t = wrapped_receiver<Receiver, Func>;
                using wrapped_op_state_t =
                    decltype(hpx::execution::experimental::connect(
                        std::declval<Sender&&>(),
                        std::declval<wrapped_receiver_t&&>()));

                wrapped_op_state_t wrapped_op_;

                operation_state(
                    Sender&& sender, Func&& func, Receiver&& receiver)
                  : wrapped_op_(
                        hpx::execution::experimental::connect(std::move(sender),
                            wrapped_receiver_t{
                                std::move(receiver), std::move(func)}))
                {
                }

                void start() noexcept
                {
                    hpx::execution::experimental::start(wrapped_op_);
                }
            };

            template <typename Receiver>
            auto connect(Receiver&& receiver) &&
            {
                return operation_state<Receiver>{std::move(sender_),
                    std::move(func_), std::forward<Receiver>(receiver)};
            }

            template <typename Receiver>
            auto connect(Receiver&& receiver) const&
            {
                return operation_state<Receiver>{
                    sender_, func_, std::forward<Receiver>(receiver)};
            }
#else
            auto get_env() const noexcept
            {
                throw std::runtime_error("stdexec not enabled");
            }

            template <typename Receiver>
            auto connect(Receiver&&) &&
            {
                throw std::runtime_error("stdexec not enabled");
            }

            template <typename Receiver>
            auto connect(Receiver&&) const&
            {
                throw std::runtime_error("stdexec not enabled");
            }
#endif
        };

        // Schedule returns parallel_sender
        friend constexpr parallel_sender<thread_pool_policy_scheduler>
        tag_invoke(hpx::execution::experimental::schedule_t,
            [[maybe_unused]] thread_pool_policy_scheduler&& sched)
        {
#ifdef HPX_HAVE_STDEXEC
            return parallel_sender<thread_pool_policy_scheduler>{
                HPX_MOVE(sched)};
#else
            throw std::runtime_error("stdexec not enabled");
#endif
        }

        friend constexpr parallel_sender<thread_pool_policy_scheduler>
        tag_invoke(hpx::execution::experimental::schedule_t,
            [[maybe_unused]] thread_pool_policy_scheduler const& sched)
        {
#ifdef HPX_HAVE_STDEXEC
            return parallel_sender<thread_pool_policy_scheduler>{sched};
#else
            throw std::runtime_error("stdexec not enabled");
#endif
        }

        // Extend operation_state to check stop token
        template <typename Scheduler, typename Receiver>
        struct parallel_operation_state
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

            template <typename Scheduler_, typename Receiver_>
            parallel_operation_state(
                Scheduler_&& scheduler, Receiver_&& receiver)
              : scheduler(HPX_FORWARD(Scheduler_, scheduler))
              , receiver(HPX_FORWARD(Receiver_, receiver))
            {
            }

            parallel_operation_state(parallel_operation_state&&) = delete;
            parallel_operation_state(parallel_operation_state const&) = delete;
            parallel_operation_state& operator=(
                parallel_operation_state&&) = delete;
            parallel_operation_state& operator=(
                parallel_operation_state const&) = delete;

            ~parallel_operation_state() = default;

            friend void tag_invoke(
                start_t, parallel_operation_state& os) noexcept
            {
#ifdef HPX_HAVE_STDEXEC
                auto stop_token = hpx::execution::experimental::get_stop_token(
                    hpx::execution::experimental::get_env(os.receiver));
                if (stop_token.stop_requested())
                {
                    hpx::execution::experimental::set_stopped(
                        HPX_MOVE(os.receiver));
                    return;
                }
#endif
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        os.scheduler.execute(
                            [receiver = HPX_MOVE(os.receiver)]() mutable {
                                hpx::execution::experimental::set_value(
                                    HPX_MOVE(receiver));
                            });
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(os.receiver), HPX_MOVE(ep));
                    });
            }
        };

        // Connect for parallel_sender
        template <typename Receiver>
        friend parallel_operation_state<thread_pool_policy_scheduler, Receiver>
        tag_invoke(connect_t, parallel_sender<thread_pool_policy_scheduler>&& s,
            Receiver&& receiver)
        {
            return {HPX_MOVE(s.scheduler), HPX_FORWARD(Receiver, receiver)};
        }

        template <typename Receiver>
        friend parallel_operation_state<thread_pool_policy_scheduler, Receiver>
        tag_invoke(connect_t, parallel_sender<thread_pool_policy_scheduler>& s,
            Receiver&& receiver)
        {
            return {s.scheduler, HPX_FORWARD(Receiver, receiver)};
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        [[nodiscard]] std::size_t get_num_cores() const
        {
            if (num_cores_ != 0)
            {
                return num_cores_;
            }

            if constexpr (std::is_same_v<Policy, launch::sync_policy>)
            {
                return 1;
            }
            else
            {
                if (policy_.get_policy() == hpx::detail::launch_policy::sync)
                {
                    return 1;
                }

                auto const* pool =
                    pool_ ? pool_ : threads::detail::get_self_or_default_pool();
                return pool->get_os_thread_count();
            }
        }

        [[nodiscard]] constexpr std::size_t get_first_core() const noexcept
        {
            return first_core_;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        hpx::threads::thread_pool_base* pool_ =
            hpx::threads::detail::get_self_or_default_pool();
        Policy policy_;
        std::size_t first_core_ = 0;
        std::size_t num_cores_ = 0;
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
        char const* annotation_ = nullptr;
#endif
        /// \endcond
    };

    // support all properties exposed by the embedded policy
    // clang-format off
    template <typename Tag, typename Policy, typename Property,
        HPX_CONCEPT_REQUIRES_(
            hpx::execution::experimental::is_scheduling_property_v<Tag>
        )>
    // clang-format on
    auto tag_invoke(Tag tag,
        thread_pool_policy_scheduler<Policy> const& scheduler, Property&& prop)
        -> decltype(std::declval<thread_pool_policy_scheduler<Policy>>().policy(
                        std::declval<Tag>()(
                            std::declval<Policy>(), std::declval<Property>())),
            thread_pool_policy_scheduler<Policy>())
    {
        auto scheduler_with_prop = scheduler;
        scheduler_with_prop.policy(
            tag(scheduler.policy(), HPX_FORWARD(Property, prop)));
        return scheduler_with_prop;
    }

    // clang-format off
    template <typename Tag, typename Policy,
        HPX_CONCEPT_REQUIRES_(
            hpx::execution::experimental::is_scheduling_property_v<Tag>
        )>
    // clang-format on
    auto tag_invoke(
        Tag tag, thread_pool_policy_scheduler<Policy> const& scheduler)
        -> decltype(std::declval<Tag>()(std::declval<Policy>()))
    {
        return tag(scheduler.policy());
    }

    using thread_pool_scheduler = thread_pool_policy_scheduler<hpx::launch>;
}    // namespace hpx::execution::experimental
