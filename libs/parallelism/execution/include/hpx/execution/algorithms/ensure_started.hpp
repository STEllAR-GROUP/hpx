//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if defined(HPX_HAVE_CXX17_STD_VARIANT)
#include <hpx/assert.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/detail/single_result.hpp>
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/synchronization/mutex.hpp>
#include <hpx/thread_support/atomic_count.hpp>
#include <hpx/type_support/pack.hpp>

#include <boost/container/small_vector.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>

namespace hpx { namespace execution { namespace experimental {
    namespace detail {
        template <typename R>
        struct error_visitor
        {
            std::decay_t<R> r;

            template <typename E>
            void operator()(E const& e)
            {
                hpx::execution::experimental::set_error(std::move(r), e);
            }
        };

        template <typename R>
        struct value_visitor
        {
            std::decay_t<R> r;

            template <typename Ts>
            void operator()(Ts const& ts)
            {
                hpx::util::invoke_fused(
                    hpx::util::bind_front(
                        hpx::execution::experimental::set_value, std::move(r)),
                    ts);
            }
        };

        template <typename S, typename Allocator>
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
                    S>::template value_types<Tuple, Variant>,
                value_types_helper>;

            template <template <typename...> class Variant>
            using error_types =
                hpx::util::detail::unique_t<hpx::util::detail::prepend_t<
                    typename hpx::execution::experimental::sender_traits<
                        S>::template error_types<Variant>,
                    std::exception_ptr>>;

            static constexpr bool sends_done = false;

            struct shared_state
            {
            private:
                struct ensure_started_receiver
                {
                    hpx::intrusive_ptr<shared_state> st;

                    template <typename E>
                        void set_error(E&& e) && noexcept
                    {
                        st->v.template emplace<error_type>(
                            error_type(std::forward<E>(e)));
                        st->set_predecessor_done();
                        st.reset();
                    }

                    void set_done() && noexcept
                    {
                        st->set_predecessor_done();
                        st.reset();
                    };

                    template <typename... Ts>
                        void set_value(Ts&&... ts) && noexcept
                    {
                        st->v.template emplace<value_type>(
                            hpx::make_tuple<>(std::forward<Ts>(ts)...));

                        st->set_predecessor_done();
                        st.reset();
                    }
                };

                using allocator_type = typename std::allocator_traits<
                    Allocator>::template rebind_alloc<shared_state>;
                allocator_type alloc;
                using mutex_type = hpx::util::spinlock;
                mutex_type mtx;
                hpx::util::atomic_count reference_count{0};
                std::atomic<bool> start_called{false};
                std::atomic<bool> predecessor_done{false};

                using operation_state_type =
                    std::decay_t<connect_result_t<S, ensure_started_receiver>>;
                operation_state_type os;

                struct done_type
                {
                };
                using value_type =
                    typename hpx::execution::experimental::sender_traits<
                        S>::template value_types<hpx::tuple, std::variant>;
                using error_type =
                    hpx::util::detail::unique_t<hpx::util::detail::prepend_t<
                        error_types<std::variant>, std::exception_ptr>>;
                std::variant<std::monostate, done_type, error_type, value_type>
                    v;

                using continuation_type =
                    hpx::util::unique_function_nonser<void()>;
                boost::container::small_vector<continuation_type, 1>
                    continuations;

            public:
                template <typename S_,
                    typename = std::enable_if_t<
                        !std::is_same<std::decay_t<S_>, shared_state>::value>>
                shared_state(S_&& s, allocator_type const& alloc)
                  : alloc(alloc)
                  , os(hpx::execution::experimental::connect(
                        std::forward<S_>(s), ensure_started_receiver{this}))
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
                template <typename R>
                struct done_error_value_visitor
                {
                    std::decay_t<R> r;

                    HPX_NORETURN void operator()(std::monostate) const
                    {
                        HPX_UNREACHABLE;
                    }

                    void operator()(done_type)
                    {
                        hpx::execution::experimental::set_done(std::move(r));
                    }

                    void operator()(error_type const& e)
                    {
                        std::visit(error_visitor<R>{std::forward<R>(r)}, e);
                    }

                    void operator()(value_type const& ts)
                    {
                        std::visit(value_visitor<R>{std::forward<R>(r)}, ts);
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
                template <typename R>
                void add_continuation(R& r) = delete;

                template <typename R>
                void add_continuation(R&& r)
                {
                    if (predecessor_done)
                    {
                        // If we read predecessor_done here it means that one of
                        // set_error/set_done/set_value has been called and
                        // values/errors have been stored into the shared state.
                        // We can trigger the continuation directly.
                        std::visit(
                            done_error_value_visitor<R>{std::move(r)}, v);
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
                                done_error_value_visitor<R>{std::move(r)}, v);
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
                            continuations.emplace_back([this,
                                                           r = std::move(r)]() {
                                std::visit(
                                    done_error_value_visitor<R>{std::move(r)},
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

            hpx::intrusive_ptr<shared_state> st;

            template <typename S_>
            ensure_started_sender(S_&& s, Allocator const& a)
            {
                using allocator_type = Allocator;
                using other_allocator = typename std::allocator_traits<
                    allocator_type>::template rebind_alloc<shared_state>;
                using allocator_traits = std::allocator_traits<other_allocator>;
                using unique_ptr = std::unique_ptr<shared_state,
                    util::allocator_deleter<other_allocator>>;

                other_allocator alloc(a);
                unique_ptr p(allocator_traits::allocate(alloc, 1),
                    hpx::util::allocator_deleter<other_allocator>{alloc});

                new (p.get()) shared_state{std::forward<S_>(s), a};
                st = p.release();

                // Eagerly start the work received until this point.
                //
                // P1897r3 says "When start is called on os2 [the operation
                // state resulting from connecting the ensure_started_sender to
                // a receiver], call execution::start(os [the operation state
                // resulting from connecting S to ensure_started_receiver])",
                // which would indicate that start should be called later.
                // However, to fulfill the promise that ensure_started actually
                // eagerly submits the sender S we call start already here.
                st->start();
            }

            ensure_started_sender(ensure_started_sender const&) = default;
            ensure_started_sender& operator=(
                ensure_started_sender const&) = default;
            ensure_started_sender(ensure_started_sender&&) = default;
            ensure_started_sender& operator=(ensure_started_sender&&) = default;

            template <typename R>
            struct operation_state
            {
                std::decay_t<R> r;
                hpx::intrusive_ptr<shared_state> st;

                template <typename R_>
                operation_state(R_&& r, hpx::intrusive_ptr<shared_state> st)
                  : r(std::forward<R_>(r))
                  , st(std::move(st))
                {
                }

                operation_state(operation_state&&) = delete;
                operation_state& operator=(operation_state&&) = delete;
                operation_state(operation_state const&) = delete;
                operation_state& operator=(operation_state const&) = delete;

                void start() & noexcept
                {
                    st->add_continuation(std::move(r));
                }
            };

            template <typename R>
            operation_state<R> connect(R&& r) &&
            {
                return {std::forward<R>(r), std::move(st)};
            }

            template <typename R>
            operation_state<R> connect(R&& r) &
            {
                return {std::forward<R>(r), st};
            }
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct ensure_started_t final
      : hpx::functional::tag_fallback<ensure_started_t>
    {
    private:
        template <typename S,
            typename Allocator = hpx::util::internal_allocator<>>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            ensure_started_t, S&& s, Allocator const& a = {})
        {
            return detail::ensure_started_sender<S, Allocator>{
                std::forward<S>(s), a};
        }

        template <typename S, typename Allocator>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            ensure_started_t, detail::ensure_started_sender<S, Allocator> s,
            Allocator const& = {})
        {
            return s;
        }

        template <typename Allocator = hpx::util::internal_allocator<>>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            ensure_started_t, Allocator const& a = {})
        {
            return detail::partial_algorithm<ensure_started_t, Allocator>{a};
        }
    } ensure_started{};
}}}    // namespace hpx::execution::experimental
#endif
