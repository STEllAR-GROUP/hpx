//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
//  Copyright (c) 2022 Chuanqiu He
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/optional.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/datastructures/variant.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/detail/single_result.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/detail/tag_priority_invoke.hpp>
#include <hpx/synchronization/condition_variable.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/type_support/meta.hpp>
#include <hpx/type_support/pack.hpp>
#include <hpx/type_support/unused.hpp>

#include <atomic>
#include <exception>
#include <mutex>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental::detail {

    struct sync_wait_with_variant_error_visitor
    {
        void operator()(std::exception_ptr ep) const
        {
            std::rethrow_exception(ep);
        }

        template <typename Error>
        void operator()(Error& error) const
        {
            throw error;
        }
    };

    template <typename Sender>
    struct sync_wait_with_variant_receiver
    {
        // value and error_types of the predecessor sender
        template <template <typename...> class Tuple,
            template <typename...> class Variant>
        using predecessor_value_types =
            value_types_of_t<Sender, empty_env, Tuple, Variant>;

        template <template <typename...> class Variant>
        using predecessor_error_types =
            error_types_of_t<Sender, empty_env, Variant>;

        // The template should compute the result type of whatever returned from
        // sync_wait_with_variant, which should be optional of the variant of
        // the tuples.
        using result_type = predecessor_value_types<hpx::tuple, hpx::variant>;

        // The type of errors to store in the variant. This in itself is a
        // variant.
        using error_type =
            hpx::util::detail::unique_t<hpx::util::detail::prepend_t<
                predecessor_error_types<hpx::variant>, std::exception_ptr>>;

        // We use a spinlock here to allow taking the lock on non-HPX threads.
        using mutex_type = hpx::spinlock;

        struct shared_state
        {
            // We use a spinlock here to allow taking the lock on non-HPX
            // threads.
            hpx::condition_variable cond_var;
            mutex_type mtx;
            std::atomic<bool> set_called = false;
            hpx::variant<hpx::monostate, error_type, result_type> value;

            void wait()
            {
                if (!set_called)
                {
                    std::unique_lock<mutex_type> l(mtx);
                    if (!set_called)
                    {
                        cond_var.wait(l);
                    }
                }
            }

            auto get_value()
            {
                if (hpx::holds_alternative<result_type>(value))
                {
                    //  wrap it into an optional
                    return hpx::make_optional(
                        hpx::get<result_type>(HPX_MOVE(value)));
                }
                else if (hpx::holds_alternative<error_type>(value))
                {
                    hpx::visit(sync_wait_with_variant_error_visitor{},
                        hpx::get<error_type>(value));
                }

                // If the variant holds a hpx::monostate set_stopped was called
                // we return an empty optional
                return hpx::optional<
                    typename single_variant<result_type>::type>();
            }
        };

        shared_state& state;

        void signal_set_called() noexcept
        {
            std::unique_lock<mutex_type> l(state.mtx);
            state.set_called = true;
            hpx::util::ignore_while_checking<decltype(l)> il(&l);
            HPX_UNUSED(il);

            state.cond_var.notify_one();
        }

        template <typename Error>
        friend void tag_invoke(set_error_t, sync_wait_with_variant_receiver&& r,
            Error&& error) noexcept
        {
            r.state.value.template emplace<error_type>(
                HPX_FORWARD(Error, error));
            r.signal_set_called();
        }

        friend void tag_invoke(
            set_stopped_t, sync_wait_with_variant_receiver&& r) noexcept
        {
            r.signal_set_called();
        }

        template <typename... Us>
        friend void tag_invoke(set_value_t, sync_wait_with_variant_receiver&& r,
            Us&&... us) noexcept
        {
            r.state.value.template emplace<result_type>(
                hpx::forward_as_tuple(HPX_FORWARD(Us, us)...));
            r.signal_set_called();
        }
    };
}    // namespace hpx::execution::experimental::detail

namespace hpx::this_thread::experimental {

    inline constexpr struct sync_wait_with_variant_t final
      : hpx::functional::detail::tag_priority<sync_wait_with_variant_t>
    {
    private:
        // clang-format off
        template <typename Sender,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_sender_v<Sender> &&
                hpx::execution::experimental::detail::
                    is_completion_scheduler_tag_invocable_v<
                        hpx::execution::experimental::set_value_t,
                        Sender, sync_wait_with_variant_t
                    >
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(
            sync_wait_with_variant_t, Sender&& sender)
        {
            auto scheduler =
                hpx::execution::experimental::get_completion_scheduler<
                    hpx::execution::experimental::set_value_t>(sender);

            return hpx::functional::tag_invoke(sync_wait_with_variant_t{},
                HPX_MOVE(scheduler), HPX_FORWARD(Sender, sender));
        }

        // clang-format off
        template <typename Sender,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_sender_v<Sender>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            sync_wait_with_variant_t, Sender&& sender)
        {
            using receiver_type = hpx::execution::experimental::detail::
                sync_wait_with_variant_receiver<Sender>;
            using state_type = typename receiver_type::shared_state;

            state_type state{};
            auto op_state = hpx::execution::experimental::connect(
                HPX_FORWARD(Sender, sender), receiver_type{state});
            hpx::execution::experimental::start(op_state);

            state.wait();
            return state.get_value();
        }

        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            sync_wait_with_variant_t)
        {
            return hpx::execution::experimental::detail::partial_algorithm<
                sync_wait_with_variant_t>{};
        }
    } sync_wait_with_variant{};
}    // namespace hpx::this_thread::experimental
