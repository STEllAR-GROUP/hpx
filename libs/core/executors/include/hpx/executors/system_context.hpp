//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/datastructures/variant.hpp>
#include <hpx/execution/algorithms/sync_wait.hpp>
#include <hpx/execution/queries/get_scheduler.hpp>
#include <hpx/execution_base/completion_scheduler.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/execution_base/get_env.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/move_only_function.hpp>
#include <hpx/iterator_support/counting_iterator.hpp>
#include <hpx/memory/intrusive_ptr.hpp>
#include <hpx/synchronization/latch.hpp>
#include <hpx/thread_support/atomic_count.hpp>
#include <hpx/type_support/meta.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <exception>
#include <type_traits>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::execution::experimental {

    namespace detail {

        // forward declaration
        class system_scheduler_base;
        class system_scheduler;
        class system_sender;

        HPX_CORE_EXPORT void intrusive_ptr_add_ref(
            system_scheduler_base* p) noexcept;
        HPX_CORE_EXPORT void intrusive_ptr_release(
            system_scheduler_base* p) noexcept;

        ///////////////////////////////////////////////////////////////////////
        class system_scheduler_base
        {
        public:
            constexpr system_scheduler_base() noexcept
              : count(0)
            {
            }

            virtual ~system_scheduler_base() = default;

            virtual std::size_t max_concurrency() const noexcept = 0;

            virtual void execute(hpx::move_only_function<void()> set_value,
                hpx::move_only_function<void(std::exception_ptr)>
                    set_error) & noexcept = 0;

            // virtual void bulk_set_value(
            //     hpx::move_only_function<void()> set_value) = 0;
            // virtual void bulk_set_error(
            //     hpx::move_only_function<void()> set_error) = 0;
            // virtual void bulk_set_stopped(
            //     hpx::move_only_function<void()> set_stopped) = 0;

        private:
            friend HPX_CORE_EXPORT void intrusive_ptr_add_ref(
                system_scheduler_base* p) noexcept;
            friend HPX_CORE_EXPORT void intrusive_ptr_release(
                system_scheduler_base* p) noexcept;

            hpx::util::atomic_count count;
        };

        // This is the sender type returned from schedule on a system_scheduler.
        //
        // This sender satisfies the following properties:
        //    - Implements the get_completion_scheduler query for the value and
        //      done channel where it returns a type that is logically a pair of
        //      an object that compares equal to itself, and a representation of
        //      delegatee scheduler that may be obtained from receivers
        //      connected with the sender.
        //    - If connected with a receiver that supports the get_stop_token
        //      query, if that stop_token is stopped, operations on which start
        //      has been called, but are not yet running (and are hence not yet
        //      guaranteed to make progress) must complete with set_done as soon
        //      as is practical.
        //    - connecting the sender and calling start() on the resulting
        //      operation state are non-blocking operations.
        class system_sender
        {
        private:
            template <typename Receiver>
            struct operation_state
            {
                hpx::intrusive_ptr<system_scheduler_base> ctx;
                HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

                friend void tag_invoke(hpx::execution::experimental::start_t,
                    operation_state& os) noexcept
                {
                    os.start();
                }

                void start() & noexcept
                {
                    auto set_value = [this]() mutable {
                        hpx::execution::experimental::set_value(
                            HPX_MOVE(receiver));
                    };

                    auto set_error = [this](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(receiver), HPX_MOVE(ep));
                    };

                    ctx->execute(HPX_MOVE(set_value), HPX_MOVE(set_error));
                }
            };

        public:
            explicit system_sender(
                hpx::intrusive_ptr<system_scheduler_base> ctx) noexcept
              : ctx(HPX_MOVE(ctx))
            {
            }

            hpx::intrusive_ptr<system_scheduler_base> const& get_context() const
            {
                return ctx;
            }

        private:
            using completion_signatures =
                hpx::execution::experimental::completion_signatures<
                    hpx::execution::experimental::set_value_t(),
                    hpx::execution::experimental::set_error_t(
                        std::exception_ptr),
                    hpx::execution::experimental::set_stopped_t()>;

            template <typename Env>
            friend auto tag_invoke(
                hpx::execution::experimental::get_completion_signatures_t,
                system_sender const&, Env) noexcept -> completion_signatures;

            // clang-format off
            template <typename Receiver,
                HPX_CONCEPT_REQUIRES_(
                    hpx::execution::experimental::is_receiver_v<Receiver>
                )>
            // clang-format on
            friend operation_state<Receiver> tag_invoke(
                hpx::execution::experimental::connect_t, system_sender const& s,
                Receiver&& r)
            {
                return {s.ctx, HPX_FORWARD(Receiver, r)};
            }

            // clang-format off
            template <typename Receiver,
                HPX_CONCEPT_REQUIRES_(
                    hpx::execution::experimental::is_receiver_v<Receiver>
                )>
            // clang-format on
            friend operation_state<Receiver> tag_invoke(
                hpx::execution::experimental::connect_t, system_sender&& s,
                Receiver&& r)
            {
                return {HPX_MOVE(s.ctx), HPX_FORWARD(Receiver, r)};
            }

        private:
            hpx::intrusive_ptr<system_scheduler_base> ctx;
        };

        // A system_scheduler is a copyable handle to a system_context. It is
        // the means through which agents are launched on a system_context. The
        // system_scheduler instance does not have to outlive work submitted to
        // it. The system_scheduler is technically implementation-defined, but
        // must be nameable.
        //
        // system_scheduler is not independely constructable, and must be
        // obtained from a system_context. It is both move and copy
        // constructable and assignable.
        //
        // A system_scheduler has reference semantics with respect to its
        // system_context. Calling any operation other than the destructor on a
        // system_scheduler after the system_context it was created from is
        // destroyed is undefined behavior, and that operation may access freed
        // memory. The system_scheduler:
        //    - satisfies the scheduler concept and implements the schedule
        //      customisation point to return an implementation-defined sender
        //      type.
        //    - implements the get_forward_progress_guarantee query to return
        //      parallel.
        //    - implements the bulk CPO to customise the bulk sender adapter
        //      such that:
        //          When execution::set_value(r, args...) is called on the
        //          created receiver, an agent is created with parallel forward
        //          progress on the underlying system_context for each i of type
        //          Shape from 0 to sh, where sh is the shape parameter to the
        //          bulk call, that calls f(i, args...).
        //
        // If the underlying system_context is unable to make progress on work
        // created through system_scheduler instances, and the sender retrieved
        // from scheduler is connected to a receiver that supports the
        // get_delegatee_scheduler query, work may be scheduled on the scheduler
        // returned by get_delegatee_scheduler at the time of the call to start,
        // or at any later point before the work completes.
        class system_scheduler
        {
        public:
            explicit system_scheduler(
                hpx::intrusive_ptr<system_scheduler_base> ctx) noexcept
              : ctx(HPX_MOVE(ctx))
            {
            }

            // Two system_schedulers compare equal if they share the same
            // underlying system_context.
            constexpr bool operator==(
                system_scheduler const& rhs) const noexcept
            {
                return ctx == rhs.ctx;
            }
            constexpr bool operator!=(
                system_scheduler const& rhs) const noexcept
            {
                return ctx != rhs.ctx;
            }

            hpx::intrusive_ptr<system_scheduler_base> const& get_context() const
            {
                return ctx;
            }

        private:
            // schedule calls on a system_scheduler are non-blocking operations.
            friend system_sender tag_invoke(
                hpx::execution::experimental::schedule_t,
                system_scheduler const& sched) noexcept
            {
                return system_sender{sched.ctx};
            }

            friend constexpr hpx::execution::experimental::
                forward_progress_guarantee
                tag_invoke(hpx::execution::experimental::
                               get_forward_progress_guarantee_t,
                    system_scheduler const&) noexcept
            {
                return hpx::execution::experimental::
                    forward_progress_guarantee::parallel;
            }

            struct scheduler_holder
            {
                void finish()
                {
                    l.count_down(1);
                }

                system_scheduler const& get_scheduler() const noexcept
                {
                    return sched;
                }

                system_scheduler const& sched;
                hpx::latch& l;
            };

            template <hpx::execution::experimental::detail::sync_wait_type Type,
                typename Sender>
            decltype(auto) sync_wait(Sender&& sender) const
            {
                hpx::latch l(2);

                using receiver_type =
                    hpx::execution::experimental::detail::sync_wait_receiver<
                        Sender, scheduler_holder, Type>;
                using state_type = typename receiver_type::shared_state;

                scheduler_holder holder{*this, l};
                state_type state{};

                auto op_state = hpx::execution::experimental::connect(
                    HPX_FORWARD(Sender, sender), receiver_type{state, holder});
                hpx::execution::experimental::start(op_state);

                l.arrive_and_wait();    // wait for the variant to be filled in

                return state.get_value();
            }

            // clang-format off
            template <typename Sender,
                HPX_CONCEPT_REQUIRES_(
                    hpx::execution::experimental::is_sender_v<Sender,
                        hpx::execution::experimental::detail::
                            sync_wait_receiver_env<system_scheduler>>
                )>
            // clang-format on
            friend decltype(auto) tag_invoke(
                hpx::this_thread::experimental::sync_wait_t,
                system_scheduler const& sched, Sender&& sender)
            {
                using hpx::execution::experimental::detail::sync_wait_type;
                return sched.sync_wait<sync_wait_type::single>(
                    HPX_FORWARD(Sender, sender));
            }

            // clang-format off
            template <typename Sender,
                HPX_CONCEPT_REQUIRES_(
                    hpx::execution::experimental::is_sender_v<Sender,
                        hpx::execution::experimental::detail::
                            sync_wait_receiver_env<system_scheduler>>
                )>
            // clang-format on
            friend decltype(auto) tag_invoke(
                hpx::this_thread::experimental::sync_wait_with_variant_t,
                system_scheduler const& sched, Sender&& sender)
            {
                using hpx::execution::experimental::detail::sync_wait_type;
                return sched.sync_wait<sync_wait_type::variant>(
                    HPX_FORWARD(Sender, sender));
            }

        private:
            hpx::intrusive_ptr<system_scheduler_base> ctx;
        };

        // clang-format off
        template <typename CPO,
            HPX_CONCEPT_REQUIRES_(
                meta::value<meta::one_of<
                    std::decay_t<CPO>,
                    hpx::execution::experimental::set_value_t,
                    hpx::execution::experimental::set_stopped_t
                >>
            )>
        // clang-format on
        system_scheduler tag_invoke(
            hpx::execution::experimental::get_completion_scheduler_t<CPO>,
            system_sender const& s) noexcept
        {
            return system_scheduler{s.get_context()};
        }
    }    // namespace detail

    inline namespace p2079 {

        // The system_context creates a view on some underlying execution
        // context supporting parallel forward progress. A system_context must
        // outlive any work launched on it.
        class HPX_CORE_EXPORT system_context
        {
        public:
            system_context();
            ~system_context();

            system_context(system_context const&) = delete;
            system_context(system_context&&) = delete;
            system_context& operator=(system_context const&) = delete;
            system_context& operator=(system_context&&) = delete;

            auto get_scheduler()
            {
                return detail::system_scheduler{ctx};
            }

            std::size_t max_concurrency() const noexcept;

        private:
            hpx::intrusive_ptr<detail::system_scheduler_base> ctx;
        };
    }    // namespace p2079
}    // namespace hpx::execution::experimental

#include <hpx/config/warnings_suffix.hpp>

#include <hpx/executors/detail/bulk_system_context.hpp>
