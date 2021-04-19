//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if defined(HPX_HAVE_CXX17_STD_VARIANT)
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/detail/single_result.hpp>
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/functional/tag_fallback_invoke.hpp>
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
                    void set_error(E&& e) noexcept
                    {
                        st->v.template emplace<error_type>(
                            error_type(std::forward<E>(e)));
                        st->set_predecessor_done();
                        st.reset();
                    }

                    void set_done() noexcept
                    {
                        st->set_predecessor_done();
                        st.reset();
                    };

                    template <typename... Ts>
                    void set_value(Ts&&... ts) noexcept
                    {
                        st->v.template emplace<value_type>(
                            hpx::make_tuple<>(std::forward<Ts>(ts)...));

                        st->set_predecessor_done();
                        st.reset();
                    }
                };

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
                shared_state(S_&& s)
                  : os(hpx::execution::experimental::connect(
                        std::forward<S_>(s), ensure_started_receiver{this}))
                {
                }

                ~shared_state()
                {
                    if (!start_called)
                    {
                        std::terminate();
                    }

                    if (--reference_count == 0)
                    {
                        delete this;
                    }
                }

            private:
                template <typename R>
                struct done_error_value_visitor
                {
                    std::decay_t<R> r;

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

                    void operator()(std::monostate)
                    {
                        std::terminate();
                    }
                };

                void set_predecessor_done()
                {
                    predecessor_done = true;

                    std::unique_lock<hpx::util::spinlock> l{mtx};
                    if (!continuations.empty())
                    {
                        l.unlock();
                        for (auto const& continuation : continuations)
                        {
                            continuation();
                        }

                        continuations.clear();
                    }
                }

            public:
                template <typename R,
                    typename =
                        std::enable_if_t<std::is_rvalue_reference_v<R&&>>>
                void add_continuation(R&& r)
                {
                    if (predecessor_done)
                    {
                        std::visit(
                            done_error_value_visitor<R>{std::move(r)}, v);
                    }
                    else
                    {
                        std::unique_lock<mutex_type> l{mtx};
                        if (predecessor_done)
                        {
                            l.unlock();
                            std::visit(
                                done_error_value_visitor<R>{std::move(r)}, v);
                        }
                        else
                        {
                            continuations.emplace_back(
                                [this,
                                    // NOLINTNEXTLINE(bugprone-move-forwarding-reference)
                                    r = std::move(r)]() {
                                    std::visit(
                                        done_error_value_visitor<R>{
                                            std::move(r)},
                                        v);
                                });
                        }
                    }
                }

                void start() noexcept
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
                    auto c = --p->reference_count;
                    if (c == 0)
                    {
                        delete p;
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

                new (p.get()) shared_state{std::forward<S_>(s)};
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

                void start() noexcept
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
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            ensure_started_t, S&& s, Allocator const& a = {})
        {
            return detail::ensure_started_sender<S, Allocator>{
                std::forward<S>(s), a};
        }

        template <typename S, typename Allocator,
            typename NewAllocator = hpx::util::internal_allocator<>>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            ensure_started_t, detail::ensure_started_sender<S, Allocator> s,
            Allocator const& = {})
        {
            return s;
        }

        template <typename Allocator = hpx::util::internal_allocator<>>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            ensure_started_t, Allocator const& a = {})
        {
            return detail::partial_algorithm<ensure_started_t, Allocator>{a};
        }
    } ensure_started{};
}}}    // namespace hpx::execution::experimental
#endif
