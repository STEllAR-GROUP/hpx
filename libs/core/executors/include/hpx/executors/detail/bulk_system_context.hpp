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
#include <hpx/execution/algorithms/bulk.hpp>
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
#include <hpx/thread_support/atomic_count.hpp>
#include <hpx/type_support/meta.hpp>
#include <hpx/type_support/pack.hpp>

#include <cstddef>
#include <exception>
#include <type_traits>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::execution::experimental { namespace detail {

    // forward declaration
    class bulk_system_scheduler_base;
    class system_scheduler;

    HPX_CORE_EXPORT void intrusive_ptr_add_ref(
        bulk_system_scheduler_base* p) noexcept;
    HPX_CORE_EXPORT void intrusive_ptr_release(
        bulk_system_scheduler_base* p) noexcept;

    ///////////////////////////////////////////////////////////////////////////
    class bulk_system_scheduler_base
    {
    public:
        constexpr bulk_system_scheduler_base() noexcept
          : count(0)
        {
        }

        virtual ~bulk_system_scheduler_base() = default;

        virtual void bulk_set_value(
            hpx::move_only_function<void()> set_value) = 0;
        virtual void bulk_set_error(
            hpx::move_only_function<void()> set_error) = 0;
        virtual void bulk_set_stopped(
            hpx::move_only_function<void()> set_stopped) = 0;

    private:
        friend HPX_CORE_EXPORT void intrusive_ptr_add_ref(
            bulk_system_scheduler_base* p) noexcept;
        friend HPX_CORE_EXPORT void intrusive_ptr_release(
            bulk_system_scheduler_base* p) noexcept;

        hpx::util::atomic_count count;
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename Sender, typename Shape, typename F>
    class bulk_system_sender
    {
    private:
        template <typename OperationState>
        struct bulk_receiver
        {
            OperationState* os;

            template <typename E>
            friend void tag_invoke(hpx::execution::experimental::set_error_t,
                bulk_receiver&& r, E&& e) noexcept
            {
                auto set_error = [this, e = HPX_FORWARD(E, e)]() mutable {
                    hpx::execution::experimental::set_error(
                        HPX_MOVE(r.op_state->receiver), HPX_FORWARD(E, e));
                };
                r.op_state->ctx->bulk_set_error(HPX_MOVE(set_error));
            }

            friend void tag_invoke(hpx::execution::experimental::set_stopped_t,
                bulk_receiver&& r) noexcept
            {
                auto set_stopped = [this]() mutable {
                    hpx::execution::experimental::set_stopped(
                        HPX_MOVE(r.op_state->receiver));
                };
                r.op_state->ctx->bulk_set_stopped(HPX_MOVE(set_stopped));
            }

            friend void tag_invoke(hpx::execution::experimental::set_value_t,
                bulk_receiver&& r) noexcept
            {
                auto set_value = [this]() mutable {
                    hpx::execution::experimental::set_value(
                        HPX_MOVE(r.op_state->receiver));
                };
                r.op_state->ctx->bulk_set_value(HPX_MOVE(set_value));
            }
        };

        template <typename Receiver>
        struct bulk_operation_state
        {
            using operation_state_type =
                hpx::execution::experimental::connect_result_t<Sender,
                    bulk_receiver<bulk_operation_state>>;

            operation_state_type op_state;

            hpx::intrusive_ptr<bulk_system_scheduler_base> ctx;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Shape> shape;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

            friend void tag_invoke(hpx::execution::experimental::start_t,
                bulk_operation_state& os) noexcept
            {
                hpx::execution::experimental::start(os.op_state);
            }

            template <typename Sender_, typename Shape_, typename F_,
                typename Receiver_>
            bulk_operation_state(
                Sender_&& sender, Shape_&& shape, F_&& f, Receiver_&& receiver)
              : op_state(hpx::execution::experimental::connect(
                    HPX_FORWARD(Sender_, sender),
                    bulk_receiver<bulk_operation_state>{this}))
              , shape(HPX_FORWARD(Shape_, shape))
              , f(HPX_FORWARD(F_, f))
              , receiver(HPX_FORWARD(Receiver_, receiver))
            {
            }
        };

    public:
        template <typename Sender_, typename Shape_, typename F_>
        explicit bulk_system_sender(
            hpx::intrusive_ptr<bulk_system_scheduler_base> ctx,
            Sender_&& sender, Shape_&& shape, F_&& f) noexcept
          : ctx(HPX_MOVE(ctx))
          , sender(HPX_FORWARD(Sender_, sender))
          , shape(HPX_FORWARD(Shape_, shape))
          , f(HPX_FORWARD(F_, f))
        {
        }

    private:
        template <typename Env>
        struct generate_completion_signatures
        {
            template <template <typename...> typename Tuple,
                template <typename...> typename Variant>
            using value_types = value_types_of_t<Sender, Env, Tuple, Variant>;

            template <template <typename...> typename Variant>
            using error_types = hpx::util::detail::unique_concat_t<
                error_types_of_t<Sender, Env, Variant>,
                Variant<std::exception_ptr>>;

            static constexpr bool sends_stopped =
                sends_stopped_of_v<Sender, Env>;
        };

        template <typename Env>
        friend auto tag_invoke(
            hpx::execution::experimental::get_completion_signatures_t,
            bulk_system_sender const&, Env)
            -> generate_completion_signatures<Env>;

        // clang-format off
        template <typename Receiver,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_receiver_v<Receiver>
            )>
        // clang-format on
        friend bulk_operation_state<Receiver> tag_invoke(
            hpx::execution::experimental::connect_t,
            bulk_system_sender const& s, Receiver&& r)
        {
            return {s.ctx, s.sender, s.shape, s.f, HPX_FORWARD(Receiver, r)};
        }

        // clang-format off
        template <typename Receiver,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_receiver_v<Receiver>
            )>
        // clang-format on
        friend bulk_operation_state<Receiver> tag_invoke(
            hpx::execution::experimental::connect_t, bulk_system_sender&& s,
            Receiver&& r)
        {
            return {HPX_MOVE(s.ctx), HPX_MOVE(s.sender), HPX_MOVE(s.shape),
                HPX_MOVE(s.f), HPX_FORWARD(Receiver, r)};
        }

        friend constexpr auto tag_invoke(
            hpx::execution::experimental::get_completion_scheduler_t<
                hpx::execution::experimental::set_value_t>,
            bulk_system_sender const& s)
        {
            return system_scheduler{s.ctx};
        }

    private:
        hpx::intrusive_ptr<bulk_system_scheduler_base> ctx;
        HPX_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
        HPX_NO_UNIQUE_ADDRESS std::decay_t<Shape> shape;
        HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;
    };

    HPX_CORE_EXPORT hpx::intrusive_ptr<bulk_system_scheduler_base>
    get_bulk_context(system_scheduler const& s);

    // clang-format off
    template <typename Sender, typename Shape, typename F,
        HPX_CONCEPT_REQUIRES_(
            !std::is_integral_v<std::decay_t<Shape>>
        )>
    // clang-format on
    auto tag_invoke(hpx::execution::experimental::bulk_t,
        system_scheduler const& s, Sender&& sender, Shape&& shape,
        F&& f) noexcept
    {
        return bulk_system_sender<Sender, Shape, F>(get_bulk_context(s),
            HPX_FORWARD(Sender, sender), HPX_FORWARD(Shape, shape),
            HPX_FORWARD(F, f));
    }

    // clang-format off
    template <typename Sender, typename Count, typename F,
        HPX_CONCEPT_REQUIRES_(
            std::is_integral_v<Count>
        )>
    // clang-format on
    auto tag_invoke(hpx::execution::experimental::bulk_t,
        system_scheduler const& s, Sender&& sender, Count const& count,
        F&& f) noexcept
    {
        return bulk_system_sender<Sender, hpx::util::counting_shape<Count>, F>(
            get_bulk_context(s), HPX_FORWARD(Sender, sender),
            hpx::util::counting_shape(count), HPX_FORWARD(F, f));
    }
}}    // namespace hpx::execution::experimental::detail

#include <hpx/config/warnings_suffix.hpp>
