//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/allocator_support/allocator_deleter.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/allocator_support/traits/is_allocator.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/detail/single_result.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/futures/detail/future_data.hpp>
#include <hpx/futures/promise.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/type_support/meta.hpp>
#include <hpx/type_support/unused.hpp>

#include <exception>
#include <memory>
#include <utility>

namespace hpx::execution::experimental {

    namespace detail {

        template <typename T, typename Allocator>
        struct make_future_receiver
        {
            hpx::intrusive_ptr<
                hpx::lcos::detail::future_data_allocator<T, Allocator>>
                data;

            friend void tag_invoke(set_error_t, make_future_receiver&& r,
                std::exception_ptr ep) noexcept
            {
                r.data->set_exception(HPX_MOVE(ep));
                r.data.reset();
            }

            friend void tag_invoke(
                set_stopped_t, make_future_receiver&&) noexcept
            {
                std::terminate();
            }

            template <typename U>
            friend void tag_invoke(
                set_value_t, make_future_receiver&& r, U&& u) noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() { r.data->set_value(HPX_FORWARD(U, u)); },
                    [&](std::exception_ptr ep) {
                        r.data->set_exception(HPX_MOVE(ep));
                    });
                r.data.reset();
            }
        };

        template <typename Allocator>
        struct make_future_receiver<void, Allocator>
        {
            hpx::intrusive_ptr<
                hpx::lcos::detail::future_data_allocator<void, Allocator>>
                data;

            friend void tag_invoke(set_error_t, make_future_receiver&& r,
                std::exception_ptr ep) noexcept
            {
                r.data->set_exception(HPX_MOVE(ep));
                r.data.reset();
            }

            friend void tag_invoke(
                set_stopped_t, make_future_receiver&&) noexcept
            {
                std::terminate();
            }

            friend void tag_invoke(
                set_value_t, make_future_receiver&& r) noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() { r.data->set_value(hpx::util::unused); },
                    [&](std::exception_ptr ep) {
                        r.data->set_exception(HPX_MOVE(ep));
                    });
                r.data.reset();
            }
        };

        template <typename T, typename Allocator, typename OperationState>
        struct future_data
          : hpx::lcos::detail::future_data_allocator<T, Allocator>
        {
            HPX_NON_COPYABLE(future_data);

            using operation_state_type = std::decay_t<OperationState>;
            using init_no_addref =
                typename hpx::lcos::detail::future_data_allocator<T,
                    Allocator>::init_no_addref;
            using other_allocator = typename std::allocator_traits<
                Allocator>::template rebind_alloc<future_data>;

            operation_state_type op_state;

            template <typename Sender>
            future_data(init_no_addref no_addref, other_allocator const& alloc,
                Sender&& sender)
              : hpx::lcos::detail::future_data_allocator<T, Allocator>(
                    no_addref, alloc)
              , op_state(hpx::execution::experimental::connect(
                    HPX_FORWARD(Sender, sender),
                    detail::make_future_receiver<T, Allocator>{this}))
            {
                hpx::execution::experimental::start(op_state);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Sender, typename Allocator>
        auto make_future(Sender&& sender, Allocator const& allocator)
        {
            using allocator_type = Allocator;

            using value_types = hpx::execution::experimental::value_types_of_t<
                std::decay_t<Sender>, hpx::execution::experimental::empty_env,
                meta::pack, meta::pack>;

            using result_type =
                std::decay_t<detail::single_result_t<value_types>>;
            using operation_state_type = hpx::util::invoke_result_t<
                hpx::execution::experimental::connect_t, Sender,
                detail::make_future_receiver<result_type, allocator_type>>;

            using shared_state = detail::future_data<result_type,
                allocator_type, operation_state_type>;
            using init_no_addref = typename shared_state::init_no_addref;
            using other_allocator = typename std::allocator_traits<
                allocator_type>::template rebind_alloc<shared_state>;
            using allocator_traits = std::allocator_traits<other_allocator>;
            using unique_ptr = std::unique_ptr<shared_state,
                util::allocator_deleter<other_allocator>>;

            other_allocator alloc(allocator);
            unique_ptr p(allocator_traits::allocate(alloc, 1),
                hpx::util::allocator_deleter<other_allocator>{alloc});

            allocator_traits::construct(alloc, p.get(), init_no_addref{}, alloc,
                HPX_FORWARD(Sender, sender));

            return hpx::traits::future_access<future<result_type>>::create(
                p.release(), false);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // execution::make_future is a sender consumer that submits the work
    // described by the provided sender for execution, similarly to
    // ensure_started, except that it returns a future that provides an optional
    // tuple of values that were sent by the provided sender on its completion
    // of work.
    //
    // Where 4.20.1 execution::schedule and 4.20.3 execution::transfer_just are
    // meant to enter the domain of senders, make_future is meant to exit the
    // domain of senders, retrieving the result of the task graph.
    //
    // If the provided sender sends an error instead of values, make_future
    // stores that error as an exception in the future, or the original
    // exception if the error is of type std::exception_ptr.
    //
    // If the provided sender sends the "stopped" signal instead of values,
    // make_future calls std::terminate.
    //
    inline constexpr struct make_future_t final
      : hpx::functional::detail::tag_fallback<make_future_t>
    {
    private:
        // clang-format off
        template <typename Sender,
            typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(make_future_t,
            Sender&& sender, Allocator const& allocator = Allocator{})
        {
            return detail::make_future(HPX_FORWARD(Sender, sender), allocator);
        }

        // clang-format off
        template <typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            make_future_t, Allocator const& allocator = Allocator{})
        {
            return detail::partial_algorithm<make_future_t, Allocator>{
                allocator};
        }
    } make_future{};
}    // namespace hpx::execution::experimental
