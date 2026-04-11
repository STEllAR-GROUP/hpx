//  Copyright (c) 2021 ETH Zurich
//  Copyright (c) 2022-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_STDEXEC)
#include <hpx/modules/execution_base.hpp>
#else

#include <hpx/assert.hpp>
#include <hpx/execution/algorithms/detail/inject_scheduler.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/detail/single_result.hpp>
#include <hpx/execution/algorithms/run_loop.hpp>
#include <hpx/modules/allocator_support.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/modules/datastructures.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/execution_base.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/tag_invoke.hpp>
#include <hpx/modules/thread_support.hpp>
#include <hpx/modules/type_support.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    namespace detail {

        HPX_CXX_CORE_EXPORT enum class submission_type { eager, lazy };

        HPX_CXX_CORE_EXPORT template <typename Receiver>
        struct error_visitor
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver>& receiver;

            template <typename Error>
            void operator()(Error const& error) noexcept
            {
                // FIXME: check whether it is ok to move the receiver
                hpx::execution::experimental::set_error(
                    HPX_MOVE(receiver), error);
            }
        };

        HPX_CXX_CORE_EXPORT template <typename Receiver>
        struct value_visitor
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver>& receiver;

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

        HPX_CXX_CORE_EXPORT template <typename Sender, typename Allocator,
            submission_type Type, typename Scheduler = no_scheduler>
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

            // clang-format off
            template <typename Env>
            friend auto tag_invoke(get_completion_signatures_t,
                split_sender const&,
                Env) -> generate_completion_signatures<Env>;

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
                        hpx::visit(error_visitor<Receiver>{receiver}, error);
                    }

                    void operator()(value_type const& ts)
                    {
                        hpx::visit(value_visitor<Receiver>{receiver}, ts);
                    }
                };

                // schedule_completion dispatches a stored continuation to
                // the correct execution context. The base implementation fires
                // the continuation inline (no scheduler attached). Subclasses
                // override this to reroute through a specific scheduler,
                // ensuring P2300 get_completion_scheduler guarantees hold for
                // late-arriving subscribers (i.e. when predecessor_done is
                // already true at the time add_continuation is called).
                virtual void schedule_completion(
                    continuation_type&& continuation)
                {
                    continuation();
                }

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
                        for (auto& continuation : continuations)
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
                        // We dispatch the completion through schedule_completion
                        // so that any attached scheduler is honoured, satisfying
                        // the P2300 get_completion_scheduler contract for late
                        // subscribers.
                        schedule_completion([this,
                                                receiver = HPX_FORWARD(Receiver,
                                                    receiver)]() mutable {
                            hpx::visit(
                                done_error_value_visitor<Receiver>{
                                    HPX_MOVE(receiver)},
                                v);
                        });
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
                            // release the lock early and dispatch through
                            // schedule_completion to honour the scheduler.
                            l.unlock();
                            schedule_completion(
                                [this,
                                    receiver = HPX_FORWARD(
                                        Receiver, receiver)]() mutable {
                                    hpx::visit(
                                        done_error_value_visitor<Receiver>{
                                            HPX_MOVE(receiver)},
                                        v);
                                });
                        }
                        else
                        {
                            // If predecessor_done is still false, we add the
                            // continuation to the vector of continuations. This
                            // has to be done while holding the lock, since
                            // other threads may also try to add continuations
                            // to the vector and the vector is not threadsafe in
                            // itself. The continuation will be called later
                            // when set_error/set_stopped/set_value is called
                            // (via set_predecessor_done).
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
                    p->reference_count.increment();
                }

                friend void intrusive_ptr_release(shared_state* p) noexcept
                {
                    if (p->reference_count.decrement() == 0)
                    {
                        // The thread that decrements the reference count to
                        // zero must perform an acquire to ensure that it
                        // doesn't start destructing the object until all
                        // previous writes have drained.
                        std::atomic_thread_fence(std::memory_order_acquire);

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

            // shared_state_scheduler wraps a generic Scheduler and overrides
            // schedule_completion so that late-arriving subscribers receive
            // their completion signal dispatched on the scheduler's execution
            // context, preserving the P2300 get_completion_scheduler contract.
            //
            // Note: this is intentionally separate from shared_state_run_loop
            // to avoid adding a run_loop dependency for general schedulers.
            template <typename Sched>
            struct shared_state_scheduler : shared_state
            {
                // Store a decay-copy of the scheduler, matching the rule that
                // tag_invoke(get_completion_scheduler_t<CPO>, split_sender)
                // returns an equivalent scheduler.
                HPX_NO_UNIQUE_ADDRESS std::decay_t<Sched> sched;

                // clang-format off
                template <typename Sender_,
                    HPX_CONCEPT_REQUIRES_(
                        meta::value<meta::none_of<
                            shared_state_scheduler, std::decay_t<Sender_>>>
                    )>
                // clang-format on
                shared_state_scheduler(Sender_&& sender,
                    typename shared_state::allocator_type const& alloc,
                    Sched&& scheduler)
                  : shared_state(HPX_FORWARD(Sender_, sender), alloc)
                  , sched(HPX_FORWARD(Sched, scheduler))
                {
                }

                ~shared_state_scheduler() override = default;

                // Dispatch the continuation through the scheduler so that the
                // downstream receiver executes on the correct execution context
                // even when predecessor_done is already true (late subscriber).
                void schedule_completion(
                    typename shared_state::continuation_type&& continuation)
                    override
                {
                    using continuation_type =
                        typename shared_state::continuation_type;
                    using base_alloc_type =
                        typename shared_state::allocator_type;

                    // Self-owning holder using HPX's allocator — mirrors exactly
                    // how shared_state and operation_state_holder manage their
                    // own lifetimes via intrusive_ptr + allocator_traits.
                    struct schedule_op_holder;

                    struct schedule_receiver
                    {
                        hpx::intrusive_ptr<schedule_op_holder> holder;

                        friend void tag_invoke(
                            set_value_t, schedule_receiver&& r) noexcept
                        {
                            // Move continuation out first; releasing holder
                            // (and destroying it) must come after the
                            // continuation has been invoked.
                            auto h = HPX_MOVE(r.holder);
                            HPX_MOVE(h->cont)();
                            // h now goes out of scope → intrusive_ptr_release
                            // → allocator destroy+deallocate.
                        }

                        template <typename Error>
                        [[noreturn]] friend void tag_invoke(
                            set_error_t, schedule_receiver&&, Error&&) noexcept
                        {
                            // schedule() must not produce errors.
                            std::terminate();
                        }

                        friend void tag_invoke(
                            set_stopped_t, schedule_receiver&& r) noexcept
                        {
                            // Scheduler stopped: drop continuation silently.
                            r.holder.reset();
                        }

                        friend empty_env tag_invoke(
                            get_env_t, schedule_receiver const&) noexcept
                        {
                            return {};
                        }
                    };

                    using schedule_sender_type = hpx::util::invoke_result_t<
                        hpx::execution::experimental::schedule_t,
                        std::decay_t<Sched>&>;
                    using op_state_type = connect_result_t<schedule_sender_type,
                        schedule_receiver>;

                    // Use the same allocator that manages shared_state's own
                    // lifetime, rebound for schedule_op_holder.
                    struct schedule_op_holder
                    {
                        using holder_alloc_type =
                            typename std::allocator_traits<base_alloc_type>::
                                template rebind_alloc<schedule_op_holder>;

                        continuation_type cont;
                        hpx::util::atomic_count ref_count{0};
                        HPX_NO_UNIQUE_ADDRESS holder_alloc_type alloc;
                        op_state_type op_state;

                        schedule_op_holder(continuation_type&& c,
                            std::decay_t<Sched>& s, holder_alloc_type const& a)
                          : cont(HPX_MOVE(c))
                          , alloc(a)
                          , op_state(hpx::execution::experimental::connect(
                                hpx::execution::experimental::schedule(s),
                                schedule_receiver{
                                    hpx::intrusive_ptr<schedule_op_holder>(
                                        this)}))
                        {
                        }

                        friend void intrusive_ptr_add_ref(
                            schedule_op_holder* p) noexcept
                        {
                            p->ref_count.increment();
                        }

                        friend void intrusive_ptr_release(
                            schedule_op_holder* p) noexcept
                        {
                            if (p->ref_count.decrement() == 0)
                            {
                                // Copy allocator out before destroying self,
                                // matching the pattern in shared_state.
                                std::atomic_thread_fence(
                                    std::memory_order_acquire);
                                holder_alloc_type a(p->alloc);
                                std::allocator_traits<
                                    holder_alloc_type>::destroy(a, p);
                                std::allocator_traits<
                                    holder_alloc_type>::deallocate(a, p, 1);
                            }
                        }
                    };

                    using holder_alloc_type =
                        typename std::allocator_traits<base_alloc_type>::
                            template rebind_alloc<schedule_op_holder>;
                    using holder_alloc_traits =
                        std::allocator_traits<holder_alloc_type>;
                    using holder_unique_ptr =
                        std::unique_ptr<schedule_op_holder,
                            util::allocator_deleter<holder_alloc_type>>;

                    // Construct the holder using the shared_state allocator.
                    // unique_ptr guards against leaks if construct() throws.
                    holder_alloc_type holder_alloc(this->alloc);
                    holder_unique_ptr p(
                        holder_alloc_traits::allocate(holder_alloc, 1),
                        hpx::util::allocator_deleter<holder_alloc_type>{
                            holder_alloc});
                    holder_alloc_traits::construct(holder_alloc, p.get(),
                        HPX_MOVE(continuation), sched, holder_alloc);

                    // Keep an owning reference while start() executes so that
                    // a synchronous set_value() cannot destroy the holder
                    // before start() returns (which would dereference freed
                    // memory via the raw op_state pointer).
                    hpx::intrusive_ptr<schedule_op_holder> owner(p.release());
                    hpx::execution::experimental::start(owner->op_state);
                    // owner goes out of scope here; if start() was synchronous
                    // the holder is destroyed now; otherwise schedule_receiver
                    // holds the last reference until set_value/set_stopped.
                }
            };

            hpx::intrusive_ptr<shared_state> state;

            template <typename Sender_, typename Scheduler_ = no_scheduler,
                typename = std::enable_if_t<
                    !is_scheduler_v<std::decay_t<Scheduler_>> ||
                    std::is_same_v<std::decay_t<Scheduler_>,
                        run_loop_scheduler>>>
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

            // Constructor for a generic (non-run_loop) scheduler: creates
            // shared_state_scheduler so that late-arriving subscribers have
            // their completions dispatched on the scheduler's context.
            // SFINAE ensures this overload only fires for schedulers that are
            // not run_loop_scheduler (which has its own explicit overload below).
            template <typename Sender_, typename Sched_,
                typename = std::enable_if_t<
                    is_scheduler_v<std::decay_t<Sched_>> &&
                    !std::is_same_v<std::decay_t<Sched_>, run_loop_scheduler>>>
            split_sender(
                Sender_&& sender, Allocator const& allocator, Sched_&& sched)
              : scheduler(HPX_FORWARD(Sched_, sched))
            {
                using sched_shared_state =
                    shared_state_scheduler<std::decay_t<Sched_>>;
                using other_allocator = typename std::allocator_traits<
                    Allocator>::template rebind_alloc<sched_shared_state>;
                using allocator_traits = std::allocator_traits<other_allocator>;
                using unique_ptr = std::unique_ptr<sched_shared_state,
                    util::allocator_deleter<other_allocator>>;

                other_allocator alloc(allocator);
                unique_ptr p(allocator_traits::allocate(alloc, 1),
                    hpx::util::allocator_deleter<other_allocator>{alloc});

                allocator_traits::construct(alloc, p.get(),
                    HPX_FORWARD(Sender_, sender), allocator, scheduler);
                state = p.release();

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
    HPX_CXX_CORE_EXPORT inline constexpr struct split_t final
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

        // Scheduler-aware split for generic (non-run_loop) schedulers.
        // Dispatches completions for late-arriving subscribers through the
        // provided scheduler, satisfying the P2300 get_completion_scheduler
        // contract. This overload is selected when passing a scheduler
        // explicitly: tag_invoke(split, my_scheduler, sender, allocator).
        // clang-format off
        template <typename Scheduler, typename Sender,
            typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                hpx::execution::experimental::is_scheduler_v<Scheduler> &&
                !std::is_same_v<std::decay_t<Scheduler>,
                    hpx::execution::experimental::run_loop_scheduler> &&
                hpx::execution::experimental::is_sender_v<Sender> &&
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_invoke(split_t,
            Scheduler&& scheduler, Sender&& sender,
            Allocator const& allocator = {})
        {
            return detail::split_sender<Sender, Allocator,
                detail::submission_type::lazy, std::decay_t<Scheduler>>{
                HPX_FORWARD(Sender, sender), allocator,
                HPX_FORWARD(Scheduler, scheduler)};
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

#endif
