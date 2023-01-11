//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/allocator_support/allocator_deleter.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/allocator_support/traits/is_allocator.hpp>
#include <hpx/assert.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/algorithms/detail/inject_scheduler.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/run_loop.hpp>
#include <hpx/execution_base/completion_scheduler.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/detail/tag_priority_invoke.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/thread_support/atomic_count.hpp>
#include <hpx/type_support/meta.hpp>
#include <hpx/type_support/unused.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    namespace detail {

        template <typename Derived, typename Sender, typename Allocator>
        struct operation_state_holder_base
        {
            struct start_detached_receiver
            {
                hpx::intrusive_ptr<Derived> op_state;

                template <typename Error>
                [[noreturn]] friend void tag_invoke(
                    set_error_t, start_detached_receiver&&, Error&&) noexcept
                {
                    HPX_ASSERT_MSG(false,
                        "set_error was called on the receiver of "
                        "start_detached, terminating. If you want to allow "
                        "errors from the predecessor sender, handle them first "
                        "with e.g. let_error.");
                    std::terminate();
                }

                friend void tag_invoke(
                    set_stopped_t, start_detached_receiver&& r) noexcept
                {
                    r.op_state->finish();
                    r.op_state.reset();
                };

                template <typename... Ts>
                friend void tag_invoke(
                    set_value_t, start_detached_receiver&& r, Ts&&...) noexcept
                {
                    r.op_state->finish();
                    r.op_state.reset();
                }
            };

        protected:
            using allocator_type = typename std::allocator_traits<
                Allocator>::template rebind_alloc<Derived>;

        private:
            HPX_NO_UNIQUE_ADDRESS allocator_type alloc;
            hpx::util::atomic_count count{0};

            using operation_state_type =
                connect_result_t<Sender, start_detached_receiver>;
            std::decay_t<operation_state_type> op_state;

        public:
            template <typename Sender_>
            explicit operation_state_holder_base(
                Sender_&& sender, allocator_type const& alloc)
              : alloc(alloc)
              , op_state(connect(HPX_FORWARD(Sender_, sender),
                    start_detached_receiver{static_cast<Derived*>(this)}))
            {
                hpx::execution::experimental::start(op_state);
            }

        private:
            friend void intrusive_ptr_add_ref(
                operation_state_holder_base* p) noexcept
            {
                ++p->count;
            }

            friend void intrusive_ptr_release(
                operation_state_holder_base* p) noexcept
            {
                if (--p->count == 0)
                {
                    allocator_type other_alloc(p->alloc);
                    std::allocator_traits<allocator_type>::destroy(
                        other_alloc, static_cast<Derived*>(p));
                    std::allocator_traits<allocator_type>::deallocate(
                        other_alloc, static_cast<Derived*>(p), 1);
                }
            }
        };

        template <typename Sender, typename Allocator>
        struct operation_state_holder
          : operation_state_holder_base<
                operation_state_holder<Sender, Allocator>, Sender, Allocator>
        {
            using base_type = operation_state_holder_base<
                operation_state_holder<Sender, Allocator>, Sender, Allocator>;
            using allocator_type = typename base_type::allocator_type;

            template <typename Sender_>
            explicit operation_state_holder(
                Sender_&& sender, allocator_type const& alloc)
              : base_type(HPX_FORWARD(Sender_, sender), alloc)
            {
            }

            static constexpr void finish() noexcept {}
        };

        template <typename Sender, typename Allocator>
        struct operation_state_holder_with_run_loop
          : operation_state_holder_base<
                operation_state_holder_with_run_loop<Sender, Allocator>, Sender,
                Allocator>
        {
        private:
            hpx::execution::experimental::run_loop& loop;

            using base_type = operation_state_holder_base<
                operation_state_holder_with_run_loop<Sender, Allocator>, Sender,
                Allocator>;

        public:
            template <typename Sender_, typename Allocator_>
            explicit operation_state_holder_with_run_loop(
                hpx::execution::experimental::run_loop_scheduler const& sched,
                Sender_&& sender, Allocator_ const& alloc)
              : base_type(HPX_FORWARD(Sender_, sender), alloc)
              , loop(sched.get_run_loop())
            {
                // keep ourselves alive
                hpx::intrusive_ptr<operation_state_holder_with_run_loop> this_(
                    this);
                loop.run();
            }

            void finish() noexcept
            {
                loop.finish();
            }
        };
    }    // namespace detail

    // execution::start_detached is used to eagerly start a sender without the
    // caller needing to manage the lifetimes of any objects.
    //
    // Like ensure_started, but does not return a value; if the provided sender
    // sends an error instead of a value, std::terminate is called.
    inline constexpr struct start_detached_t final
      : hpx::functional::detail::tag_priority<start_detached_t>
    {
    private:
        // clang-format off
        template <typename Sender,
            typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                hpx::traits::is_allocator_v<Allocator> &&
                experimental::detail::is_completion_scheduler_tag_invocable_v<
                    hpx::execution::experimental::set_value_t,
                    start_detached_t, Sender, Allocator
                >
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(
            start_detached_t, Sender&& sender,
            Allocator const& allocator = Allocator{})
        {
            auto scheduler = get_completion_scheduler<
                hpx::execution::experimental::set_value_t>(sender);

            return hpx::functional::tag_invoke(start_detached_t{},
                HPX_MOVE(scheduler), HPX_FORWARD(Sender, sender), allocator);
        }

        // clang-format off
        template <typename Sender,
            typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_sender_v<Sender> &&
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_invoke(start_detached_t,
            hpx::execution::experimental::run_loop_scheduler const& sched,
            Sender&& sender, Allocator const& allocator = {})
        {
            using allocator_type = Allocator;
            using operation_state_type =
                detail::operation_state_holder_with_run_loop<Sender, Allocator>;
            using other_allocator = typename std::allocator_traits<
                allocator_type>::template rebind_alloc<operation_state_type>;
            using allocator_traits = std::allocator_traits<other_allocator>;
            using unique_ptr = std::unique_ptr<operation_state_type,
                util::allocator_deleter<other_allocator>>;

            other_allocator alloc(allocator);
            unique_ptr p(allocator_traits::allocate(alloc, 1),
                hpx::util::allocator_deleter<other_allocator>{alloc});

            allocator_traits::construct(
                alloc, p.get(), sched, HPX_FORWARD(Sender, sender), alloc);
            HPX_UNUSED(p.release());
        }

        // clang-format off
        template <typename Sender,
            typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE void tag_fallback_invoke(
            start_detached_t, Sender&& sender,
            Allocator const& allocator = Allocator{})
        {
            using allocator_type = Allocator;
            using operation_state_type =
                detail::operation_state_holder<Sender, Allocator>;
            using other_allocator = typename std::allocator_traits<
                allocator_type>::template rebind_alloc<operation_state_type>;
            using allocator_traits = std::allocator_traits<other_allocator>;
            using unique_ptr = std::unique_ptr<operation_state_type,
                util::allocator_deleter<other_allocator>>;

            other_allocator alloc(allocator);
            unique_ptr p(allocator_traits::allocate(alloc, 1),
                hpx::util::allocator_deleter<other_allocator>{alloc});

            allocator_traits::construct(
                alloc, p.get(), HPX_FORWARD(Sender, sender), alloc);
            HPX_UNUSED(p.release());
        }

        // clang-format off
        template <typename Scheduler, typename Allocator,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_scheduler_v<Scheduler> &&
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            start_detached_t, Scheduler&& scheduler,
            Allocator const& allocator = {})
        {
            return hpx::execution::experimental::detail::inject_scheduler<
                start_detached_t, Scheduler, Allocator>{
                HPX_FORWARD(Scheduler, scheduler), allocator};
        }

        // clang-format off
        template <typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            start_detached_t, Allocator const& allocator = Allocator{})
        {
            return detail::partial_algorithm<start_detached_t, Allocator>{
                allocator};
        }
    } start_detached{};
}    // namespace hpx::execution::experimental
