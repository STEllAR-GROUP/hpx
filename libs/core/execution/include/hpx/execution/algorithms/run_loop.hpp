//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/queries/get_scheduler.hpp>
#include <hpx/execution/queries/get_stop_token.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/execution_base/get_env.hpp>
#include <hpx/modules/lock_registration.hpp>
#include <hpx/synchronization/condition_variable.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/type_support/meta.hpp>
#include <hpx/type_support/unused.hpp>

#include <exception>
#include <mutex>

namespace hpx::execution::experimental {

    // A run_loop is an execution context on which work can be scheduled. It
    // maintains a simple, thread-safe first-in-first-out queue of work. Its
    // run() member function removes elements from the queue and executes them
    // in a loop on whatever thread of execution calls run().
    //
    // A run_loop instance has an associated count that corresponds to the
    // number of work items that are in its queue. Additionally, a run_loop has
    // an associated state that can be one of starting, running, or finishing.
    //
    // Concurrent invocations of the member functions of run_loop, other than
    // run and its destructor, do not introduce data races. The member functions
    // pop_front, push_back, and finish execute atomically.
    //
    // [Note: Implementations are encouraged to use an intrusive queue of
    // operation states to hold the work units to make scheduling
    // allocation-free. -- end note]
    //
    class run_loop
    {
        struct run_loop_opstate_base
        {
            explicit run_loop_opstate_base(run_loop_opstate_base* tail) noexcept
              : next(this)
              , tail(tail)
            {
            }

            run_loop_opstate_base(run_loop_opstate_base* tail,
                void (*execute)(run_loop_opstate_base*) noexcept) noexcept
              : next(tail)
              , execute_(execute)
            {
            }

            run_loop_opstate_base(run_loop_opstate_base&&) = delete;
            run_loop_opstate_base& operator=(run_loop_opstate_base&&) = delete;

            run_loop_opstate_base* next;
            union
            {
                void (*execute_)(run_loop_opstate_base*) noexcept;
                run_loop_opstate_base* tail;
            };

            void execute() noexcept
            {
                (*execute_)(this);
            }
        };

        template <typename ReceiverId>
        struct run_loop_opstate
        {
            using Receiver = hpx::meta::type<ReceiverId>;

            struct type : run_loop_opstate_base
            {
                using id = run_loop_opstate;

                run_loop& loop;
                HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

                template <typename Receiver_>
                type(run_loop_opstate_base* tail, run_loop& loop,
                    Receiver_&& receiver) noexcept(noexcept(std::
                        is_nothrow_constructible_v<std::decay_t<Receiver>,
                            Receiver_>))
                  : run_loop_opstate_base(tail)
                  , loop(loop)
                  , receiver(HPX_FORWARD(Receiver_, receiver))
                {
                }

                static void execute(run_loop_opstate_base* p) noexcept
                {
                    auto& receiver = static_cast<type*>(p)->receiver;
                    hpx::detail::try_catch_exception_ptr(
                        [&]() {
                            if (get_stop_token(get_env(receiver))
                                    .stop_requested())
                            {
                                hpx::execution::experimental::set_stopped(
                                    HPX_MOVE(receiver));
                            }
                            else
                            {
                                hpx::execution::experimental::set_value(
                                    HPX_MOVE(receiver));
                            }
                        },
                        [&](std::exception_ptr ep) {
                            hpx::execution::experimental::set_error(
                                HPX_MOVE(receiver), HPX_MOVE(ep));
                        });
                }

                explicit type(run_loop_opstate_base* tail) noexcept
                  : run_loop_opstate_base(tail)
                {
                }

                type(run_loop_opstate_base* next, run_loop& loop, Receiver r)
                  : run_loop_opstate_base(next, &execute)
                  , loop(loop)
                  , receiver(HPX_MOVE(r))
                {
                }

                friend void tag_invoke(
                    hpx::execution::experimental::start_t, type& os) noexcept
                {
                    os.start();
                }

                void start() & noexcept;
            };
        };

    public:
        class run_loop_scheduler
        {
        public:
            using type = run_loop_scheduler;
            using id = run_loop_scheduler;

            struct run_loop_sender
            {
                using is_sender = void;
                using type = run_loop_sender;
                using id = run_loop_sender;

                explicit run_loop_sender(run_loop& loop) noexcept
                  : loop(loop)
                {
                }

            private:
                friend run_loop_scheduler;

                template <typename Receiver>
                using operation_state = hpx::meta::type<
                    run_loop_opstate<hpx::meta::get_id_t<Receiver>>>;

                template <typename Receiver>
                friend operation_state<Receiver> tag_invoke(
                    hpx::execution::experimental::connect_t,
                    run_loop_sender const& s,
                    Receiver&& receiver) noexcept(noexcept(std::
                        is_nothrow_constructible_v<operation_state<Receiver>,
                            run_loop_opstate_base*, run_loop&>))
                {
                    return operation_state<Receiver>(
                        &s.loop.head, s.loop, HPX_FORWARD(Receiver, receiver));
                }

                // clang-format off
                template <typename CPO,
                    HPX_CONCEPT_REQUIRES_(
                        meta::value<meta::one_of<
                            std::decay_t<CPO>, set_value_t, set_stopped_t>>
                    )>
                // clang-format on
                friend run_loop_scheduler tag_invoke(
                    hpx::execution::experimental::get_completion_scheduler_t<
                        CPO>,
                    run_loop_sender const& s) noexcept
                {
                    return run_loop_scheduler{s.loop};
                }

                using completion_signatures =
                    hpx::execution::experimental::completion_signatures<
                        hpx::execution::experimental::set_value_t(),
                        hpx::execution::experimental::set_error_t(
                            std::exception_ptr),
                        hpx::execution::experimental::set_stopped_t()>;

                template <typename Env>
                friend auto tag_invoke(
                    hpx::execution::experimental::get_completion_signatures_t,
                    run_loop_sender const&, Env) noexcept
                    -> completion_signatures;

                run_loop& loop;
            };

            friend run_loop;

        public:
            explicit run_loop_scheduler(run_loop& loop) noexcept
              : loop(loop)
            {
            }

            run_loop& get_run_loop() const noexcept
            {
                return loop;
            }

        private:
            friend run_loop_sender tag_invoke(
                hpx::execution::experimental::schedule_t,
                run_loop_scheduler const& sched) noexcept
            {
                return run_loop_sender(sched.loop);
            }

            friend constexpr hpx::execution::experimental::
                forward_progress_guarantee
                tag_invoke(hpx::execution::experimental::
                               get_forward_progress_guarantee_t,
                    run_loop_scheduler const&) noexcept
            {
                return hpx::execution::experimental::
                    forward_progress_guarantee::parallel;
            }

            friend constexpr bool operator==(run_loop_scheduler const& lhs,
                run_loop_scheduler const& rhs) noexcept
            {
                return &lhs.loop == &rhs.loop;
            }
            friend constexpr bool operator!=(run_loop_scheduler const& lhs,
                run_loop_scheduler const& rhs) noexcept
            {
                return !(lhs == rhs);
            }

        private:
            run_loop& loop;
        };

    private:
        friend struct run_loop_scheduler::run_loop_sender;

        hpx::spinlock mtx;
        hpx::condition_variable cond_var;

        // MSVC and gcc don't properly handle the friend declaration above
#if defined(HPX_MSVC) || defined(HPX_GCC_VERSION)
    public:
#endif
        run_loop_opstate_base head;

    private:
        bool stop = false;

        void push_back(run_loop_opstate_base* t)
        {
            std::unique_lock l(mtx);
            stop = false;
            t->next = &head;
            head.tail = head.tail->next = t;
            cond_var.notify_one();
        }

        run_loop_opstate_base* pop_front()
        {
            std::unique_lock l(mtx);
            cond_var.wait(l, [this] { return head.next != &head || stop; });
            if (head.tail == head.next)
            {
                head.tail = &head;
            }

            // std::exchange(head.next, head.next->next);
            auto old_val = HPX_MOVE(head.next);
            head.next = HPX_MOVE(head.next->next);
            return old_val;
        }

    public:
        // [exec.run_loop.ctor] construct/copy/destroy
        run_loop() noexcept
          : head(&head)    //-V546
        {
        }

        run_loop(run_loop&&) = delete;
        run_loop& operator=(run_loop&&) = delete;

        // If count is not 0 or if state is running, invokes terminate().
        // Otherwise, has no effects.
        ~run_loop()
        {
            if (head.next != &head || !stop)
            {
                std::terminate();
            }
        }

        // [exec.run_loop.members] Member functions:
        run_loop_scheduler get_scheduler()
        {
            return run_loop_scheduler(*this);
        }

        void run()
        {
            // Precondition: state is starting.
            //HPX_ASSERT(head.next != &head || !stop);
            for (run_loop_opstate_base* t; (t = pop_front()) != &head; /**/)
            {
                t->execute();
            }
            HPX_ASSERT(stop);    // Postcondition: state is finishing.
        }

        void finish()
        {
            std::unique_lock l(mtx);
            hpx::util::ignore_while_checking<decltype(l)> il(&l);
            HPX_UNUSED(il);
            stop = true;
            cond_var.notify_all();
        }
    };

    using run_loop_scheduler = run_loop::run_loop_scheduler;

    ///////////////////////////////////////////////////////////////////////////
    template <typename ReceiverId>
    inline void run_loop::run_loop_opstate<ReceiverId>::type::start() & noexcept
    try
    {
        loop.push_back(this);
    }
    catch (...)
    {
        set_error(HPX_MOVE(receiver), std::current_exception());
    }
}    // namespace hpx::execution::experimental
