//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2022-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/allocator_support/allocator_deleter.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/allocator_support/traits/is_allocator.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/algorithms/detail/inject_scheduler.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/detail/single_result.hpp>
#include <hpx/execution/algorithms/run_loop.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/detail/tag_priority_invoke.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/futures/detail/future_data.hpp>
#include <hpx/futures/promise.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/type_support/meta.hpp>
#include <hpx/type_support/unused.hpp>

#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    // enforce proper formatting
    namespace detail {

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
        template <typename Data>
        struct future_receiver_base
        {
            using is_receiver = void;
            hpx::intrusive_ptr<Data> data;

        protected:
            // Note: MSVC toolset v142 fails compiling 'data->' below, using
            // '(*data).' instead
            template <typename U>
            void set_value(U&& u) && noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() { (*data).set_value(HPX_FORWARD(U, u)); },
                    [&](std::exception_ptr ep) {
                        (*data).set_exception(HPX_MOVE(ep));
                    });
                data.reset();
            }

        private:
            friend void tag_invoke(set_error_t, future_receiver_base&& r,
                std::exception_ptr ep) noexcept
            {
                r.data->set_exception(HPX_MOVE(ep));
                r.data.reset();
            }

            friend void tag_invoke(
                set_stopped_t, future_receiver_base&&) noexcept
            {
                std::terminate();
            }
        };
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic pop
#endif

        template <typename T>
        struct future_receiver
          : future_receiver_base<hpx::lcos::detail::future_data_base<T>>
        {
        private:
            template <typename U>
            friend void tag_invoke(
                set_value_t, future_receiver&& r, U&& u) noexcept
            {
                HPX_MOVE(r).set_value(HPX_FORWARD(U, u));
            }
        };

        template <>
        struct future_receiver<void>
          : future_receiver_base<hpx::lcos::detail::future_data_base<void>>
        {
        private:
            friend void tag_invoke(set_value_t, future_receiver&& r) noexcept
            {
                HPX_MOVE(r).set_value(hpx::util::unused);
            }
        };

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
        template <typename T, typename Allocator, typename OperationState,
            typename Derived = void>
        struct future_data
          : hpx::lcos::detail::future_data_allocator<T, Allocator,
                std::conditional_t<std::is_void_v<Derived>,
                    future_data<T, Allocator, OperationState, Derived>,
                    Derived>>
        {
            future_data& operator=(future_data const&) = delete;
            future_data& operator=(future_data&&) = delete;

            using derived_type = std::conditional_t<std::is_void_v<Derived>,
                future_data, Derived>;
            using base_type = hpx::lcos::detail::future_data_allocator<T,
                Allocator, derived_type>;
            using operation_state_type = std::decay_t<OperationState>;
            using init_no_addref = typename base_type::init_no_addref;
            using other_allocator = typename std::allocator_traits<
                Allocator>::template rebind_alloc<future_data>;

            operation_state_type op_state;

            // NOLINTBEGIN(bugprone-crtp-constructor-accessibility)
            future_data(future_data const&) = delete;
            future_data(future_data&&) = delete;

            template <typename Sender>
            future_data(init_no_addref no_addref, other_allocator const& alloc,
                Sender&& sender)
              : base_type(no_addref, alloc)
              , op_state(hpx::execution::experimental::connect(
                    HPX_FORWARD(Sender, sender),
                    detail::future_receiver<T>{{this}}))
            {
                hpx::execution::experimental::start(op_state);
            }
            // NOLINTEND(bugprone-crtp-constructor-accessibility)
        };

        template <typename T, typename Allocator, typename OperationState>
        struct future_data_with_run_loop
          : future_data<T, Allocator, OperationState,
                future_data_with_run_loop<T, Allocator, OperationState>>
        {
            hpx::execution::experimental::run_loop& loop;

            using base_type = future_data<T, Allocator, OperationState,
                future_data_with_run_loop>;
            using init_no_addref = typename base_type::init_no_addref;
            using other_allocator = typename base_type::other_allocator;

            template <typename Sender>
            future_data_with_run_loop(init_no_addref no_addref,
                other_allocator const& alloc,
#if defined(HPX_HAVE_STDEXEC)
                decltype(std::declval<hpx::execution::experimental::run_loop>()
                        .get_scheduler()) const& sched,
#else
                hpx::execution::experimental::run_loop_scheduler const& sched,
#endif
                Sender&& sender)
              : base_type(no_addref, alloc, HPX_FORWARD(Sender, sender))
#if defined(HPX_HAVE_STDEXEC)
              //TODO: Keep an eye on this, it is based on the internal impl of
              // stdexec, so it is subect to change. This is currently relying
              // on the env struct to expose __loop_ as a public member.
              , loop(*hpx::execution::experimental::get_env(schedule(sched))
                        .__loop_)
#else
              , loop(sched.get_run_loop())
#endif
            {
                this->set_on_completed([this]() { loop.finish(); });
            }

            hpx::util::unused_type* get_result_void(
                error_code& ec = throws) override
            {
                execute_deferred(ec);
                return this->base_type::get_result_void(ec);
            }

            void execute_deferred(error_code& = throws) override
            {
                loop.run();
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
                detail::future_receiver<result_type>>;

            using shared_state =
                future_data<result_type, allocator_type, operation_state_type>;
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
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 110000
#pragma GCC diagnostic pop
#endif

        ///////////////////////////////////////////////////////////////////////
        template <typename Sender, typename Allocator>
        auto make_future_with_run_loop(
#if defined(HPX_HAVE_STDEXEC)
            decltype(std::declval<hpx::execution::experimental::run_loop>()
                    .get_scheduler()) const& sched,
#else
            hpx::execution::experimental::run_loop_scheduler const& sched,
#endif
            Sender&& sender, Allocator const& allocator)
        {
            using allocator_type = Allocator;

            using value_types = hpx::execution::experimental::value_types_of_t<
                std::decay_t<Sender>, hpx::execution::experimental::empty_env,
                meta::pack, meta::pack>;

            using result_type =
                std::decay_t<detail::single_result_t<value_types>>;
            using operation_state_type = hpx::util::invoke_result_t<
                hpx::execution::experimental::connect_t, Sender,
                detail::future_receiver<result_type>>;

            using shared_state = future_data_with_run_loop<result_type,
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
                sched, HPX_FORWARD(Sender, sender));

            return hpx::traits::future_access<future<result_type>>::create(
                p.release(), false);
        }
    }    // namespace detail
}    // namespace hpx::execution::experimental

namespace hpx::traits::detail {

    template <typename T, typename Allocator, typename OperationState,
        typename NewAllocator>
    struct shared_state_allocator<hpx::execution::experimental::detail::
                                      future_data<T, Allocator, OperationState>,
        NewAllocator>
    {
        using type = hpx::execution::experimental::detail::future_data<T,
            NewAllocator, OperationState>;
    };

    template <typename T, typename Allocator, typename OperationState,
        typename NewAllocator>
    struct shared_state_allocator<
        hpx::execution::experimental::detail::future_data_with_run_loop<T,
            Allocator, OperationState>,
        NewAllocator>
    {
        using type =
            hpx::execution::experimental::detail::future_data_with_run_loop<T,
                NewAllocator, OperationState>;
    };
}    // namespace hpx::traits::detail

namespace hpx::execution::experimental {

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
      : hpx::functional::detail::tag_priority<make_future_t>
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
                    Sender, make_future_t, Allocator
                >
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(make_future_t,
            Sender&& sender, Allocator const& allocator = Allocator{})
        {
#if defined(HPX_HAVE_STDEXEC)
            auto scheduler =
                hpx::execution::experimental::get_completion_scheduler<
                    hpx::execution::experimental::set_value_t>(
                    hpx::execution::experimental::get_env(sender));
#else
            auto scheduler =
                hpx::execution::experimental::get_completion_scheduler<
                    hpx::execution::experimental::set_value_t>(sender);
#endif

            return hpx::functional::tag_invoke(make_future_t{},
                HPX_MOVE(scheduler), HPX_FORWARD(Sender, sender), allocator);
        }

        // clang-format off
        template <typename Sender,
            typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_sender_v<Sender>
            )>
        // clang-format on
        friend auto tag_invoke(make_future_t,
#if defined(HPX_HAVE_STDEXEC)
            decltype(std::declval<hpx::execution::experimental::run_loop>()
                    .get_scheduler()) const& sched,
#else
            hpx::execution::experimental::run_loop_scheduler const& sched,
#endif
            Sender&& sender, Allocator const& allocator = Allocator{})
        {
            return detail::make_future_with_run_loop(
                sched, HPX_FORWARD(Sender, sender), allocator);
        }

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
        template <typename Scheduler,
            typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_scheduler_v<Scheduler> &&
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(make_future_t,
            Scheduler&& scheduler, Allocator const& allocator = Allocator{})
        {
            return hpx::execution::experimental::detail::inject_scheduler<
                make_future_t, Scheduler, Allocator>{
                HPX_FORWARD(Scheduler, scheduler), allocator};
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
