//  Copyright (c) 2021 ETH Zurich
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
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/futures/detail/future_data.hpp>
#include <hpx/futures/promise.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/type_support/pack.hpp>
#include <hpx/type_support/unused.hpp>

#include <exception>
#include <memory>
#include <utility>

namespace hpx { namespace execution { namespace experimental {
    namespace detail {
        template <typename T, typename Allocator>
        struct make_future_receiver
        {
            hpx::intrusive_ptr<
                hpx::lcos::detail::future_data_allocator<T, Allocator>>
                data;

            friend void tag_dispatch(set_error_t, make_future_receiver&& r,
                std::exception_ptr ep) noexcept
            {
                r.data->set_exception(std::move(ep));
                r.data.reset();
            }

            friend void tag_dispatch(
                set_done_t, make_future_receiver&&) noexcept
            {
                std::terminate();
            }

            template <typename U>
            friend void tag_dispatch(
                set_value_t, make_future_receiver&& r, U&& u) noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() { r.data->set_value(std::forward<U>(u)); },
                    [&](std::exception_ptr ep) {
                        r.data->set_exception(std::move(ep));
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

            friend void tag_dispatch(set_error_t, make_future_receiver&& r,
                std::exception_ptr ep) noexcept
            {
                r.data->set_exception(std::move(ep));
                r.data.reset();
            }

            friend void tag_dispatch(
                set_done_t, make_future_receiver&&) noexcept
            {
                std::terminate();
            }

            friend void tag_dispatch(
                set_value_t, make_future_receiver&& r) noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() { r.data->set_value(hpx::util::unused); },
                    [&](std::exception_ptr ep) {
                        r.data->set_exception(std::move(ep));
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
                    std::forward<Sender>(sender),
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

            using value_types =
                typename hpx::execution::experimental::sender_traits<
                    std::decay_t<Sender>>::template value_types<hpx::util::pack,
                    hpx::util::pack>;
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
                std::forward<Sender>(sender));

            return hpx::traits::future_access<future<result_type>>::create(
                p.release(), false);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    HPX_INLINE_CONSTEXPR_VARIABLE struct make_future_t final
      : hpx::functional::tag_fallback<make_future_t>
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
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            make_future_t, Sender&& sender,
            Allocator const& allocator = Allocator{})
        {
            return detail::make_future(std::forward<Sender>(sender), allocator);
        }

        // clang-format off
        template <typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            make_future_t, Allocator const& allocator = Allocator{})
        {
            return detail::partial_algorithm<make_future_t, Allocator>{
                allocator};
        }
    } make_future{};
}}}    // namespace hpx::execution::experimental
