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
#include <hpx/datastructures/detail/small_vector.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/datastructures/variant.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/algorithms/detail/inject_scheduler.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/detail/single_result.hpp>
#include <hpx/execution/algorithms/run_loop.hpp>
#include <hpx/execution_base/completion_scheduler.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/detail/tag_priority_invoke.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/functional/move_only_function.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/thread_support/atomic_count.hpp>
#include <hpx/type_support/meta.hpp>
#include <hpx/type_support/pack.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    namespace detail {

        enum class submission_type
        {
            eager,
            lazy
        };

        template <typename Receiver>
        struct error_visitor
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

            template <typename Error>
            void operator()(Error const& error) noexcept
            {
                // FIXME: check whether it is ok to move the receiver
                hpx::execution::experimental::set_error(
                    HPX_MOVE(receiver), error);
            }
        };

        template <typename Receiver>
        struct value_visitor
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

            template <typename Ts>
            void operator()(Ts const& ts) noexcept
            {
                // FIXME: check whether it is ok to move the receiver
                hpx::invoke_fused(
                    hpx::bind_front(hpx::execution::experimental::set_value,
                        HPX_MOVE(receiver)),
                    ts);
            }
        };

        template <typename Sender, typename Allocator, submission_type Type,
            typename Scheduler = no_scheduler>
        struct split_sender
        {
            using is_sender = void;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;

            template <typename Tuple>
            struct value_types_helper
            {
                using const_type =
                    hpx::util::detail::transform_t<Tuple, std::add_const>;
                using type = hpx::util::detail::transform_t<const_type,
                    std::add_lvalue_reference>;
            };

            template <typename Env>
            struct generate_completion_signatures
            {
                template <template <typename...> typename Tuple,
                    template <typename...> typename Variant>
                using value_types = hpx::util::detail::transform_t<
                    value_types_of_t<Sender, Env, Tuple, Variant>,
                    value_types_helper>;

                template <template <typename...> typename Variant>
                using error_types =
                    hpx::util::detail::unique_t<hpx::util::detail::prepend_t<
                        error_types_of_t<Sender, Env, Variant>,
                        std::exception_ptr>>;

                static constexpr bool sends_stopped = true;
            };

            template <typename Env>
            friend auto tag_invoke(
                get_completion_signatures_t, split_sender const&, Env)
                -> generate_completion_signatures<Env>;

            // clang-format off
            template <typename CPO, typename Scheduler_ = Scheduler,
                HPX_CONCEPT_REQUIRES_(
                    hpx::execution::experimental::is_scheduler_v<Scheduler_> &&
                    is_receiver_cpo_v<std::decay_t<CPO>>
                )>
            // clang-format on
            friend constexpr auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<CPO>,
                split_sender const& sender)
            {
                return sender.scheduler;
            }

            // TODO: add forwarding_sender_query

            struct shared_state
            {
                struct split_receiver;

                using allocator_type = typename std::allocator_traits<
                    Allocator>::template rebind_alloc<shared_state>;
                HPX_NO_UNIQUE_ADDRESS allocator_type alloc;

                hpx::spinlock mtx;
                hpx::util::atomic_count reference_count{0};
                std::atomic<bool> start_called{false};
                std::atomic<bool> predecessor_done{false};

                using operation_state_type =
                    std::decay_t<connect_result_t<Sender, split_receiver>>;
                operation_state_type os;

                using signatures = generate_completion_signatures<empty_env>;

                struct stopped_type
                {
                };
                using value_type = value_types_of_t<Sender, empty_env,
                    decayed_tuple, hpx::variant>;
                using error_type = detail::error_types_from<signatures,
                    meta::func<hpx::variant>>;

                hpx::variant<hpx::monostate, stopped_type, error_type,
                    value_type>
                    v;

                using continuation_type = hpx::move_only_function<void()>;
                hpx::detail::small_vector<continuation_type, 1> continuations;

                struct split_receiver
                {
                    hpx::intrusive_ptr<shared_state> state;

                    template <typename Error>
                    friend void tag_invoke(
                        set_error_t, split_receiver&& r, Error&& error) noexcept
                    {
                        HPX_MOVE(r).set_error(HPX_FORWARD(Error, error));
                    }

                    template <typename Error>
                    void set_error(Error&& error) && noexcept
                    {
                        try
                        {
                            state->v.template emplace<error_type>(
                                error_type(HPX_FORWARD(Error, error)));
                        }
                        catch (...)
                        {
                            // no way of reporting this error
                            std::terminate();
                        }
                        state->set_predecessor_done();
                        state.reset();
                    }

                    friend void tag_invoke(
                        set_stopped_t, split_receiver&& r) noexcept
                    {
                        if (r.state)
                        {
                            r.state->v.template emplace<stopped_type>();
                            r.state->set_predecessor_done();
                            r.state.reset();
                        }
                    };

                    // This typedef is duplicated from the parent struct. The
                    // parent typedef is not instantiated early enough for use
                    // here.
                    using value_type = value_types_of_t<Sender, empty_env,
                        decayed_tuple, hpx::variant>;

                    // different versions of clang-format disagree
                    // clang-format off
                    template <typename... Ts>
                    friend auto tag_invoke(
                        set_value_t, split_receiver&& r, Ts&&... ts) noexcept
                        -> decltype(
                            std::declval<
                                hpx::variant<hpx::monostate, value_type>>()
                                .template emplace<value_type>(
                                    hpx::tuple<std::decay_t<Ts>...>(
                                        HPX_FORWARD(Ts, ts)...)),
                            void())
                    // clang-format on
                    {
                        hpx::detail::try_catch_exception_ptr(
                            [&]() {
                                r.state->v.template emplace<value_type>(
                                    hpx::make_tuple(HPX_FORWARD(Ts, ts)...));
                                r.state->set_predecessor_done();
                                r.state.reset();
                            },
                            [&](std::exception_ptr ep) {
                                HPX_MOVE(r).set_error(HPX_MOVE(ep));
                            });
                    }
                };

                // clang-format off
                template <typename Sender_,
                    HPX_CONCEPT_REQUIRES_(
                        meta::value<meta::none_of<
                            shared_state, std::decay_t<Sender_>>>
                    )>
                // clang-format on
                shared_state(Sender_&& sender, allocator_type const& alloc)
                  : alloc(alloc)
                  , os(hpx::execution::experimental::connect(
                        HPX_FORWARD(Sender_, sender), split_receiver{this}))
                {
                }

                virtual ~shared_state()
                {
                    HPX_ASSERT_MSG(start_called,
                        "start was never called on the operation state of "
                        "split or ensure_started. Did you forget to connect the"
                        "sender to a receiver, or call start on the operation "
                        "state?");
                }

                template <typename Receiver>
                struct done_error_value_visitor
                {
                    HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

                    [[noreturn]] void operator()(hpx::monostate) const
                    {
                        HPX_UNREACHABLE;
                    }

                    void operator()(stopped_type)
                    {
                        hpx::execution::experimental::set_stopped(
                            HPX_MOVE(receiver));
                    }

                    void operator()(error_type const& error)
                    {
                        hpx::visit(error_visitor<Receiver>{HPX_FORWARD(
                                       Receiver, receiver)},
                            error);
                    }

                    void operator()(value_type const& ts)
                    {
                        hpx::visit(value_visitor<Receiver>{HPX_FORWARD(
                                       Receiver, receiver)},
                            ts);
                    }
                };

                virtual void set_predecessor_done()
                {
                    predecessor_done = true;

                    {
                        // We require taking the lock here to synchronize with
                        // threads attempting to add continuations to the vector
                        // of continuations. However, it is enough to take it
                        // once and release it immediately.
                        //
                        // Without the lock we may not see writes to the vector.
                        // With the lock threads attempting to add continuations
                        // will either:
                        // - See predecessor_done = true in which case they will
                        //   call the continuation directly without adding it to
                        //   the vector of continuations. Accessing the vector
                        //   below without the lock is safe in this case because
                        //   the vector is not modified.
                        // - See predecessor_done = false and proceed to take
                        //   the lock. If they see predecessor_done after taking
                        //   the lock they can again release the lock and call
                        //   the continuation directly. Accessing the vector
                        //   without the lock is again safe because the vector
                        //   is not modified.
                        // - See predecessor_done = false and proceed to take
                        //   the lock. If they see predecessor_done is still
                        //   false after taking the lock, they will proceed to
                        //   add a continuation to the vector. Since they keep
                        //   the lock they can safely write to the vector. This
                        //   thread will not proceed past the lock until they
                        //   have finished writing to the vector.
                        //
                        // Importantly, once this thread has taken and released
                        // this lock, threads attempting to add continuations to
                        // the vector must see predecessor_done = true after
                        // taking the lock in their threads and will not add
                        // continuations to the vector.
                        std::unique_lock l{mtx};
                    }

                    if (!continuations.empty())
                    {
                        for (auto const& continuation : continuations)
                        {
                            continuation();
                        }

                        continuations.clear();
                    }
                }

                template <typename Receiver>
                void add_continuation(Receiver& receiver) = delete;

                template <typename Receiver>
                void add_continuation(Receiver&& receiver)
                {
                    if (predecessor_done)
                    {
                        // If we read predecessor_done here it means that one of
                        // set_error/set_stopped/set_value has been called and
                        // values/errors have been stored into the shared state.
                        // We can trigger the continuation directly.
                        // TODO: Should this preserve the scheduler? It does not
                        // if we call set_* inline.
                        hpx::visit(
                            done_error_value_visitor<Receiver>{
                                HPX_FORWARD(Receiver, receiver)},
                            v);
                    }
                    else
                    {
                        // If predecessor_done is false, we have to take the
                        // lock to potentially add the continuation to the
                        // vector of continuations.
                        std::unique_lock l{mtx};

                        if (predecessor_done)
                        {
                            // By the time the lock has been taken,
                            // predecessor_done might already be true and we can
                            // release the lock early and call the continuation
                            // directly again.
                            l.unlock();
                            hpx::visit(
                                done_error_value_visitor<Receiver>{
                                    HPX_FORWARD(Receiver, receiver)},
                                v);
                        }
                        else
                        {
                            // If predecessor_done is still false, we add the
                            // continuation to the vector of continuations. This
                            // has to be done while holding the lock, since
                            // other threads may also try to add continuations
                            // to the vector and the vector is not threadsafe in
                            // itself. The continuation will be called later
                            // when set_error/set_stopped/set_value is called.
                            continuations.emplace_back(
                                [this,
                                    receiver = HPX_FORWARD(
                                        Receiver, receiver)]() mutable {
                                    hpx::visit(
                                        done_error_value_visitor<Receiver>{
                                            HPX_MOVE(receiver)},
                                        v);
                                });
                        }
                    }
                }

                void start() & noexcept
                {
                    if (!start_called.exchange(true))
                    {
                        hpx::execution::experimental::start(os);
                    }
                }

                friend void intrusive_ptr_add_ref(shared_state* p) noexcept
                {
                    ++p->reference_count;
                }

                friend void intrusive_ptr_release(shared_state* p) noexcept
                {
                    if (--p->reference_count == 0)
                    {
                        allocator_type other_alloc(p->alloc);
                        std::allocator_traits<allocator_type>::destroy(
                            other_alloc, p);
                        std::allocator_traits<allocator_type>::deallocate(
                            other_alloc, p, 1);
                    }
                }
            };

            struct shared_state_run_loop : shared_state
            {
                run_loop& loop;

                // clang-format off
                template <typename Sender_,
                    HPX_CONCEPT_REQUIRES_(
                        meta::value<meta::none_of<
                            shared_state_run_loop, std::decay<Sender_>>>
                    )>
                // clang-format on
                shared_state_run_loop(Sender_&& sender,
                    typename shared_state::allocator_type const& alloc,
                    run_loop& loop)
                  : shared_state(HPX_FORWARD(Sender_, sender), alloc)
                  , loop(loop)
                {
                }

                ~shared_state_run_loop() override = default;

                void set_predecessor_done() override
                {
                    shared_state::set_predecessor_done();
                    loop.finish();
                }
            };

            hpx::intrusive_ptr<shared_state> state;

            template <typename Sender_, typename Scheduler_ = no_scheduler>
            split_sender(Sender_&& sender, Allocator const& allocator,
                Scheduler_&& scheduler = Scheduler_{})
              : scheduler(HPX_FORWARD(Scheduler_, scheduler))
            {
                using allocator_type = Allocator;
                using other_allocator = typename std::allocator_traits<
                    allocator_type>::template rebind_alloc<shared_state>;
                using allocator_traits = std::allocator_traits<other_allocator>;
                using unique_ptr = std::unique_ptr<shared_state,
                    util::allocator_deleter<other_allocator>>;

                other_allocator alloc(allocator);
                unique_ptr p(allocator_traits::allocate(alloc, 1),
                    hpx::util::allocator_deleter<other_allocator>{alloc});

                allocator_traits::construct(
                    alloc, p.get(), HPX_FORWARD(Sender_, sender), allocator);
                state = p.release();

                // Eager submission means that we start the predecessor
                // operation state already when creating the sender. We don't
                // wait for another receiver to be connected.
                if constexpr (Type == submission_type::eager)
                {
                    state->start();
                }
            }

            template <typename Sender_>
            split_sender(Sender_&& sender, Allocator const& allocator,
                run_loop_scheduler const& sched)
              : scheduler(sched)
            {
                using allocator_type = Allocator;
                using other_allocator =
                    typename std::allocator_traits<allocator_type>::
                        template rebind_alloc<shared_state_run_loop>;
                using allocator_traits = std::allocator_traits<other_allocator>;
                using unique_ptr = std::unique_ptr<shared_state_run_loop,
                    util::allocator_deleter<other_allocator>>;

                other_allocator alloc(allocator);
                unique_ptr p(allocator_traits::allocate(alloc, 1),
                    hpx::util::allocator_deleter<other_allocator>{alloc});

                allocator_traits::construct(alloc, p.get(),
                    HPX_FORWARD(Sender_, sender), allocator,
                    sched.get_run_loop());
                state = p.release();

                // Eager submission means that we start the predecessor
                // operation state already when creating the sender. We don't
                // wait for another receiver to be connected.
                if constexpr (Type == submission_type::eager)
                {
                    state->start();
                }
            }

            split_sender(split_sender const&) = default;
            split_sender& operator=(split_sender const&) = default;
            split_sender(split_sender&&) = default;
            split_sender& operator=(split_sender&&) = default;

            template <typename Receiver>
            struct operation_state
            {
                HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
                hpx::intrusive_ptr<shared_state> state;

                template <typename Receiver_>
                operation_state(Receiver_&& receiver,
                    hpx::intrusive_ptr<shared_state> state)
                  : receiver(HPX_FORWARD(Receiver_, receiver))
                  , state(HPX_MOVE(state))
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                friend void tag_invoke(start_t, operation_state& os) noexcept
                {
                    // Lazy submission means that we wait to start the
                    // predecessor operation state when a downstream operation
                    // state is started, i.e. this start function is called.
                    if constexpr (Type == submission_type::lazy)
                    {
                        os.state->start();
                    }

                    os.state->add_continuation(HPX_MOVE(os.receiver));
                }
            };

            template <typename Receiver>
            friend operation_state<Receiver> tag_invoke(
                connect_t, split_sender&& s, Receiver&& receiver)
            {
                return {HPX_FORWARD(Receiver, receiver), HPX_MOVE(s.state)};
            }

            template <typename Receiver>
            friend operation_state<Receiver> tag_invoke(
                connect_t, split_sender& s, Receiver&& receiver)
            {
                return {HPX_FORWARD(Receiver, receiver), s.state};
            }
        };
    }    // namespace detail

    // execution::split is used to adapt an arbitrary sender into a sender that
    // can be connected multiple times.
    //
    // If the provided sender is a multi-shot sender, returns that sender.
    // Otherwise, returns a multi-shot sender which sends values equivalent to
    // the values sent by the provided sender.
    //
    // A single-shot sender can only be connected to a receiver at most once.
    // Its implementation of execution::connect only has overloads for an
    // rvalue-qualified sender. Callers must pass the sender as an rvalue to the
    // call to execution::connect, indicating that the call consumes the sender.
    //
    // A multi-shot sender can be connected to multiple receivers and can be
    // launched multiple times. Multi-shot senders customise execution::connect
    // to accept an lvalue reference to the sender. Callers can indicate that
    // they want the sender to remain valid after the call to execution::connect
    // by passing an lvalue reference to the sender to call these overloads.
    // Multi-shot senders should also define overloads of execution::connect
    // that accept rvalue-qualified senders to allow the sender to be also used
    // in places where only a single-shot sender is required.
    inline constexpr struct split_t final
      : hpx::functional::detail::tag_priority<split_t>
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
                    Sender, split_t, Allocator
                >
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_override_invoke(
            split_t, Sender&& sender, Allocator const& allocator = {})
        {
            auto scheduler =
                hpx::execution::experimental::get_completion_scheduler<
                    hpx::execution::experimental::set_value_t>(sender);

            return hpx::functional::tag_invoke(split_t{}, HPX_MOVE(scheduler),
                HPX_FORWARD(Sender, sender), allocator);
        }

        // clang-format off
        template <typename Sender,
            typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_sender_v<Sender> &&
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_invoke(split_t,
            hpx::execution::experimental::run_loop_scheduler const& sched,
            Sender&& sender, Allocator const& allocator = {})
        {
            return detail::split_sender<Sender, Allocator,
                detail::submission_type::lazy,
                hpx::execution::experimental::run_loop_scheduler>{
                HPX_FORWARD(Sender, sender), allocator, sched};
        }

        // clang-format off
        template <typename Sender,
            typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_sender_v<Sender> &&
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            split_t, Sender&& sender, Allocator const& allocator = {})
        {
            return detail::split_sender<Sender, Allocator,
                detail::submission_type::lazy>{
                HPX_FORWARD(Sender, sender), allocator};
        }

        template <typename Sender, typename Allocator>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(split_t,
            detail::split_sender<Sender, Allocator,
                detail::submission_type::lazy>
                sender,
            Allocator const& = {})
        {
            return sender;
        }

        // clang-format off
        template <typename Scheduler, typename Allocator,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_scheduler_v<Scheduler> &&
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            split_t, Scheduler&& scheduler, Allocator const& allocator = {})
        {
            return hpx::execution::experimental::detail::inject_scheduler<
                split_t, Scheduler, Allocator>{
                HPX_FORWARD(Scheduler, scheduler), allocator};
        }

        // clang-format off
        template <typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            split_t, Allocator const& allocator = {})
        {
            return detail::partial_algorithm<split_t, Allocator>{allocator};
        }
    } split{};
}    // namespace hpx::execution::experimental
