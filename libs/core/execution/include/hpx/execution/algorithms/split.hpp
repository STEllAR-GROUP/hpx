//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/allocator_support/allocator_deleter.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/allocator_support/traits/is_allocator.hpp>
#include <hpx/assert.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/datastructures/variant.hpp>
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

namespace hpx { namespace execution { namespace experimental {
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
            void operator()(Error const& error)
            {
                hpx::execution::experimental::set_error(
                    std::move(receiver), error);
            }
        };

        template <typename Receiver>
        struct value_visitor
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

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

        template <typename Sender, typename Allocator, submission_type Type>
        struct split_sender
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
                struct split_receiver;

                using allocator_type = typename std::allocator_traits<
                    Allocator>::template rebind_alloc<shared_state>;
                HPX_NO_UNIQUE_ADDRESS allocator_type alloc;
                using mutex_type = hpx::lcos::local::spinlock;
                mutex_type mtx;
                hpx::util::atomic_count reference_count{0};
                std::atomic<bool> start_called{false};
                std::atomic<bool> predecessor_done{false};

                using operation_state_type =
                    std::decay_t<connect_result_t<Sender, split_receiver>>;
                operation_state_type os;

                struct done_type
                {
                };
                using value_type =
                    typename hpx::execution::experimental::sender_traits<
                        Sender>::template value_types<hpx::tuple, hpx::variant>;
                using error_type =
                    hpx::util::detail::unique_t<hpx::util::detail::prepend_t<
                        error_types<hpx::variant>, std::exception_ptr>>;
                hpx::variant<hpx::monostate, done_type, error_type, value_type>
                    v;

                using continuation_type =
                    hpx::util::unique_function_nonser<void()>;
                boost::container::small_vector<continuation_type, 1>
                    continuations;

                struct split_receiver
                {
                    hpx::intrusive_ptr<shared_state> state;

                    template <typename Error>
                    friend void tag_dispatch(
                        set_error_t, split_receiver&& r, Error&& error) noexcept
                    {
                        r.state->v.template emplace<error_type>(
                            error_type(std::forward<Error>(error)));
                        r.state->set_predecessor_done();
                        r.state.reset();
                    }

                    friend void tag_dispatch(
                        set_done_t, split_receiver&& r) noexcept
                    {
                        r.state->set_predecessor_done();
                        r.state.reset();
                    };

                    // This typedef is duplicated from the parent struct. The
                    // parent typedef is not instantiated early enough for use
                    // here.
                    using value_type =
                        typename hpx::execution::experimental::sender_traits<
                            Sender>::template value_types<hpx::tuple,
                            hpx::variant>;

                    template <typename... Ts>
                    friend auto tag_dispatch(
                        set_value_t, split_receiver&& r, Ts&&... ts) noexcept
                        -> decltype(
                            std::declval<
                                hpx::variant<hpx::monostate, value_type>>()
                                .template emplace<value_type>(
                                    hpx::make_tuple<>(std::forward<Ts>(ts)...)),
                            void())
                    {
                        r.state->v.template emplace<value_type>(
                            hpx::make_tuple<>(std::forward<Ts>(ts)...));

                        r.state->set_predecessor_done();
                        r.state.reset();
                    }
                };

                template <typename Sender_,
                    typename = std::enable_if_t<!std::is_same<
                        std::decay_t<Sender_>, shared_state>::value>>
                shared_state(Sender_&& sender, allocator_type const& alloc)
                  : alloc(alloc)
                  , os(hpx::execution::experimental::connect(
                        std::forward<Sender_>(sender), split_receiver{this}))
                {
                }

                ~shared_state()
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

                    HPX_NORETURN void operator()(hpx::monostate) const
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
                        hpx::visit(
                            error_visitor<Receiver>{
                                std::forward<Receiver>(receiver)},
                            error);
                    }

                    void operator()(value_type const& ts)
                    {
                        hpx::visit(
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
                        // TODO: Should this preserve the scheduler? It does not
                        // if we call set_* inline.
                        hpx::visit(
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
                            hpx::visit(
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
                                    hpx::visit(
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
            split_sender(Sender_&& sender, Allocator const& allocator)
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
                  : receiver(std::forward<Receiver_>(receiver))
                  , state(std::move(state))
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                friend void tag_dispatch(start_t, operation_state& os) noexcept
                {
                    // Lazy submission means that we wait to start the
                    // predecessor operation state when a downstream operation
                    // state is started, i.e. this start function is called.
                    if constexpr (Type == submission_type::lazy)
                    {
                        os.state->start();
                    }

                    os.state->add_continuation(std::move(os.receiver));
                }
            };

            template <typename Receiver>
            friend operation_state<Receiver> tag_dispatch(
                connect_t, split_sender&& s, Receiver&& receiver)
            {
                return {std::forward<Receiver>(receiver), std::move(s.state)};
            }

            template <typename Receiver>
            friend operation_state<Receiver> tag_dispatch(
                connect_t, split_sender& s, Receiver&& receiver)
            {
                return {std::forward<Receiver>(receiver), s.state};
            }
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct split_t final
      : hpx::functional::tag_fallback<split_t>
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
            split_t, Sender&& sender, Allocator const& allocator = {})
        {
            return detail::split_sender<Sender, Allocator,
                detail::submission_type::lazy>{
                std::forward<Sender>(sender), allocator};
        }

        template <typename Sender, typename Allocator>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(split_t,
            detail::split_sender<Sender, Allocator,
                detail::submission_type::lazy>
                sender,
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
            split_t, Allocator const& allocator = {})
        {
            return detail::partial_algorithm<split_t, Allocator>{allocator};
        }
    } split{};
}}}    // namespace hpx::execution::experimental
