//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/allocator_support/allocator_deleter.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
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

            void set_error(std::exception_ptr e) noexcept
            {
                data->set_exception(e);
                data.reset();
            }

            void set_done() noexcept
            {
                std::terminate();
            }

            template <typename U>
            void set_value(U&& u) noexcept
            {
                data->set_value(std::forward<U>(u));
                data.reset();
            }
        };

        template <typename Allocator>
        struct make_future_receiver<void, Allocator>
        {
            hpx::intrusive_ptr<
                hpx::lcos::detail::future_data_allocator<void, Allocator>>
                data;

            void set_error(std::exception_ptr e) noexcept
            {
                data->set_exception(e);
                data.reset();
            }

            void set_done() noexcept
            {
                std::terminate();
            }

            void set_value() noexcept
            {
                data->set_value(hpx::util::unused);
                data.reset();
            }
        };

        template <typename T, typename Allocator, typename OS>
        struct future_data
          : hpx::lcos::detail::future_data_allocator<T, Allocator>
        {
            HPX_NON_COPYABLE(future_data);

            using operation_state_type = std::decay_t<OS>;
            using init_no_addref =
                typename hpx::lcos::detail::future_data_allocator<T,
                    Allocator>::init_no_addref;
            using other_allocator = typename std::allocator_traits<
                Allocator>::template rebind_alloc<future_data>;

            operation_state_type os;

            template <typename S>
            future_data(
                init_no_addref no_addref, other_allocator const& alloc, S&& s)
              : hpx::lcos::detail::future_data_allocator<T, Allocator>(
                    no_addref, alloc)
              , os(hpx::execution::experimental::connect(std::forward<S>(s),
                    detail::make_future_receiver<T, Allocator>{this}))
            {
                hpx::execution::experimental::start(os);
            }
        };
    }    // namespace detail

    template <typename S, typename Allocator = hpx::util::internal_allocator<>,
        typename = std::enable_if_t<is_sender_v<S>>>
    auto make_future(S&& s, Allocator const& a = Allocator{})
    {
        using allocator_type = Allocator;

        using value_types =
            typename hpx::execution::experimental::sender_traits<std::decay_t<
                S>>::template value_types<hpx::util::pack, hpx::util::pack>;
        using result_type = std::decay_t<detail::single_result_t<value_types>>;
        using operation_state_type = typename hpx::util::invoke_result<
            hpx::execution::experimental::connect_t, S,
            detail::make_future_receiver<result_type, allocator_type>>::type;

        using shared_state = detail::future_data<result_type, allocator_type,
            operation_state_type>;
        using init_no_addref = typename shared_state::init_no_addref;
        using other_allocator = typename std::allocator_traits<
            allocator_type>::template rebind_alloc<shared_state>;
        using allocator_traits = std::allocator_traits<other_allocator>;
        using unique_ptr = std::unique_ptr<shared_state,
            util::allocator_deleter<other_allocator>>;

        other_allocator alloc(a);
        unique_ptr p(allocator_traits::allocate(alloc, 1),
            hpx::util::allocator_deleter<other_allocator>{alloc});

        allocator_traits::construct(
            alloc, p.get(), init_no_addref{}, alloc, std::forward<S>(s));

        return hpx::traits::future_access<future<result_type>>::create(
            p.release(), false);
    }
}}}    // namespace hpx::execution::experimental
