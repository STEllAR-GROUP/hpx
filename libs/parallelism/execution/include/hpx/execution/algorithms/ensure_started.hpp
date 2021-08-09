//  Copyright (c) 2021 ETH Zurich
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
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/detail/single_result.hpp>
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/thread_support/atomic_count.hpp>
#include <hpx/type_support/pack.hpp>

#include <boost/container/small_vector.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <variant>

namespace hpx { namespace execution { namespace experimental {
    namespace detail {
        template <typename Receiver>
        struct error_visitor
        {
            std::decay_t<Receiver> receiver;

            template <typename Error>
            void operator()(Error const& error)
            {
                hpx::execution::experimental::set_error(
                    std::move(receiver), error);
            }
        };

        template <typename Receiver>
        struct value_visitor
        {
            std::decay_t<Receiver> receiver;

            template <typename Ts>
            void operator()(Ts const& ts)
            {
                hpx::util::invoke_fused(
                    hpx::util::bind_front(
                        hpx::execution::experimental::set_value,
                        std::move(receiver)),
                    ts);
            }
        };

        template <typename Sender, typename Allocator>
        struct ensure_started_sender
        {
            template <typename Tuple>
            struct value_types_helper
            {
                using const_type =
                    hpx::util::detail::transform_t<Tuple, std::add_const>;
                using type = hpx::util::detail::transform_t<const_type,
                    std::add_lvalue_reference>;
            };

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types = hpx::util::detail::transform_t<
                typename hpx::execution::experimental::sender_traits<
                    Sender>::template value_types<Tuple, Variant>,
                value_types_helper>;

            template <template <typename...> class Variant>
            using error_types =
                hpx::util::detail::unique_t<hpx::util::detail::prepend_t<
                    typename hpx::execution::experimental::sender_traits<
                        Sender>::template error_types<Variant>,
                    std::exception_ptr>>;

            static constexpr bool sends_done = false;

            struct shared_state
            {
            private:
                struct ensure_started_receiver;

                using allocator_type = typename std::allocator_traits<
                    Allocator>::template rebind_alloc<shared_state>;
                allocator_type alloc;
                using mutex_type = hpx::lcos::local::spinlock;
                mutex_type mtx;
                hpx::util::atomic_count reference_count{0};
                std::atomic<bool> start_called{false};
                std::atomic<bool> predecessor_done{false};

                using operation_state_type = std::decay_t<
                    connect_result_t<Sender, ensure_started_receiver>>;
                operation_state_type os;

                struct done_type
                {
                };
                using value_type =
                    typename hpx::execution::experimental::sender_traits<
                        Sender>::template value_types<hpx::tuple, std::variant>;
                using error_type =
                    hpx::util::detail::unique_t<hpx::util::detail::prepend_t<
                        error_types<std::variant>, std::exception_ptr>>;
                std::variant<std::monostate, done_type, error_type, value_type>
                    v;

                using continuation_type =
                    hpx::util::unique_function_nonser<void()>;
                boost::container::small_vector<continuation_type, 1>
                    continuations;

                struct ensure_started_receiver
                {
                    hpx::intrusive_ptr<shared_state> state;

                    template <typename Error>
                    void set_error(Error&& error) && noexcept
                    {
                        state->v.template emplace<error_type>(
                            error_type(std::forward<Error>(error)));
                        state->set_predecessor_done();
                        state.reset();
                    }

                    void set_done() && noexcept
                    {
                        state->set_predecessor_done();
                        state.reset();
                    };

                    // This typedef is duplicated from the parent struct. The
                    // parent typedef is not instantiated early enough for use
                    // here.
                    using value_type =
                        typename hpx::execution::experimental::sender_traits<
                            Sender>::template value_types<hpx::tuple,
                            std::variant>;

                    template <typename... Ts>
                    auto set_value(Ts&&... ts) && noexcept -> decltype(
                        std::declval<std::variant<std::monostate, value_type>>()
                            .template emplace<value_type>(
                                hpx::make_tuple<>(std::forward<Ts>(ts)...)),
                        void())
                    {
                        state->v.template emplace<value_type>(
                            hpx::make_tuple<>(std::forward<Ts>(ts)...));

                        state->set_predecessor_done();
                        state.reset();
                    }
                };

            public:
                template <typename Sender_,
                    typename = std::enable_if_t<!std::is_same<
                        std::decay_t<Sender_>, shared_state>::value>>
                shared_state(Sender_&& sender, allocator_type const& alloc)
                  : alloc(alloc)
                  , os(hpx::execution::experimental::connect(
                        std::forward<Sender_>(sender),
                        ensure_started_receiver{this}))
                {
                }

                ~shared_state()
                {
                    HPX_ASSERT_MSG(start_called,
                        "start was never called on the operation state of "
                        "ensure_started. Did you forget to connect the sender "
                        "to a receiver, or call start on the operation state?");
                }

            private:
                template <typename Receiver>
                struct done_error_value_visitor
                {
                    std::decay_t<Receiver> receiver;

                    HPX_NORETURN void operator()(std::monostate) const
                    {
                        HPX_UNREACHABLE;
                    }

                    void operator()(done_type)
                    {
                        hpx::execution::experimental::set_done(
                            std::move(receiver));
                    }

                    void operator()(error_type const& error)
                    {
                        std::visit(
                            error_visitor<Receiver>{
                                std::forward<Receiver>(receiver)},
                            error);
                    }

                    void operator()(value_type const& ts)
                    {
                        std::visit(
                            value_visitor<Receiver>{
                                std::forward<Receiver>(receiver)},
                            ts);
                    }
                };

                void set_predecessor_done()
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
                        std::unique_lock<mutex_type> l{mtx};
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

            public:
                template <typename Receiver>
                void add_continuation(Receiver& receiver) = delete;

                template <typename Receiver>
                void add_continuation(Receiver&& receiver)
                {
                    if (predecessor_done)
                    {
                        // If we read predecessor_done here it means that one of
                        // set_error/set_done/set_value has been called and
                        // values/errors have been stored into the shared state.
                        // We can trigger the continuation directly.
                        std::visit(
                            done_error_value_visitor<Receiver>{
                                std::forward<Receiver>(receiver)},
                            v);
                    }
                    else
                    {
                        // If predecessor_done is false, we have to take the
                        // lock to potentially add the continuation to the
                        // vector of continuations.
                        std::unique_lock<mutex_type> l{mtx};

                        if (predecessor_done)
                        {
                            // By the time the lock has been taken,
                            // predecessor_done might already be true and we can
                            // release the lock early and call the continuation
                            // directly again.
                            l.unlock();
                            std::visit(
                                done_error_value_visitor<Receiver>{
                                    std::forward<Receiver>(receiver)},
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
                            // when set_error/set_done/set_value is called.
                            continuations.emplace_back(
                                [this,
                                    receiver =
                                        std::forward<Receiver>(receiver)]() {
                                    std::visit(
                                        done_error_value_visitor<Receiver>{
                                            std::move(receiver)},
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

                friend void intrusive_ptr_add_ref(shared_state* p)
                {
                    ++p->reference_count;
                }

                friend void intrusive_ptr_release(shared_state* p)
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

            hpx::intrusive_ptr<shared_state> state;

            template <typename Sender_>
            ensure_started_sender(Sender_&& sender, Allocator const& allocator)
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

                new (p.get())
                    shared_state{std::forward<Sender_>(sender), allocator};
                state = p.release();

                // Eagerly start the work received until this point.
                //
                // P1897r3 says "When start is called on os2 [the operation
                // state resulting from connecting the ensure_started_sender to
                // a receiver], call execution::start(os [the operation state
                // resulting from connecting Sender to
                // ensure_started_receiver])", which would indicate that start
                // should be called later.  However, to fulfill the promise that
                // ensure_started actually eagerly submits the sender Sender we
                // call start already here.
                state->start();
            }

            ensure_started_sender(ensure_started_sender const&) = default;
            ensure_started_sender& operator=(
                ensure_started_sender const&) = default;
            ensure_started_sender(ensure_started_sender&&) = default;
            ensure_started_sender& operator=(ensure_started_sender&&) = default;

            template <typename Receiver>
            struct operation_state
            {
                std::decay_t<Receiver> receiver;
                hpx::intrusive_ptr<shared_state> state;

                template <typename Receiver_>
                operation_state(Receiver_&& receiver,
                    hpx::intrusive_ptr<shared_state> state)
                  : receiver(std::forward<Receiver_>(receiver))
                  , state(std::move(state))
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                void start() & noexcept
                {
                    state->add_continuation(std::move(receiver));
                }
            };

            template <typename Receiver>
            operation_state<Receiver> connect(Receiver&& receiver) &&
            {
                return {std::forward<Receiver>(receiver), std::move(state)};
            }

            template <typename Receiver>
            operation_state<Receiver> connect(Receiver&& receiver) &
            {
                return {std::forward<Receiver>(receiver), state};
            }
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct ensure_started_t final
      : hpx::functional::tag_fallback<ensure_started_t>
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
            ensure_started_t, Sender&& sender, Allocator const& allocator = {})
        {
            return detail::ensure_started_sender<Sender, Allocator>{
                std::forward<Sender>(sender), allocator};
        }

        template <typename Sender, typename Allocator>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            ensure_started_t,
            detail::ensure_started_sender<Sender, Allocator> sender,
            Allocator const& = {})
        {
            return sender;
        }

        // clang-format off
        template <typename Allocator = hpx::util::internal_allocator<>,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_allocator_v<Allocator>
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            ensure_started_t, Allocator const& allocator = {})
        {
            return detail::partial_algorithm<ensure_started_t, Allocator>{
                allocator};
        }
    } ensure_started{};
}}}    // namespace hpx::execution::experimental
