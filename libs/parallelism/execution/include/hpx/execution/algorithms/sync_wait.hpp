//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if defined(HPX_HAVE_CXX17_STD_VARIANT)
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/detail/single_result.hpp>
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/synchronization/condition_variable.hpp>
#include <hpx/synchronization/mutex.hpp>
#include <hpx/type_support/pack.hpp>

#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace hpx { namespace execution { namespace experimental {
    namespace detail {
        struct sync_wait_error_visitor
        {
            void operator()(std::exception_ptr e) const
            {
                std::rethrow_exception(e);
            }

            template <typename E>
            void operator()(E& e) const
            {
                throw e;
            }
        };

        template <typename S>
        struct sync_wait_receiver
        {
            // value and error_types of the predecessor sender
            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using predecessor_value_types =
                typename hpx::execution::experimental::sender_traits<
                    S>::template value_types<Tuple, Variant>;

            template <template <typename...> class Variant>
            using predecessor_error_types =
                typename hpx::execution::experimental::sender_traits<
                    S>::template error_types<Variant>;

            // The type of the single void or non-void result that we store. If
            // there are multiple variants or multiple values sync_wait will
            // fail to compile.
            using result_type = std::decay_t<single_result_t<
                predecessor_value_types<hpx::util::pack, hpx::util::pack>>>;

            // Constant to indicate if the type of the result from the
            // predecessor sender is void or not
            static constexpr bool is_void_result =
                std::is_same_v<result_type, void>;

            // Dummy type to indicate that set_value with void has been called
            struct void_value_type
            {
            };

            // The type of the value to store in the variant, void_value_type if
            // result_type is void, or result_type if it is not
            using value_type = std::conditional_t<is_void_result,
                void_value_type, result_type>;

            // The type of errors to store in the variant. This in itself is a
            // variant.
            using error_type =
                hpx::util::detail::unique_t<hpx::util::detail::prepend_t<
                    predecessor_error_types<std::variant>, std::exception_ptr>>;

            struct state
            {
                hpx::lcos::local::condition_variable cv;
                hpx::lcos::local::mutex m;
                bool set_called = false;
                std::variant<std::monostate, error_type, value_type> v;

                void wait()
                {
                    {
                        std::unique_lock<hpx::lcos::local::mutex> l(m);
                        if (!set_called)
                        {
                            cv.wait(l);
                        }
                    }
                }

                auto get_value()
                {
                    if (std::holds_alternative<value_type>(v))
                    {
                        if constexpr (is_void_result)
                        {
                            return;
                        }
                        else
                        {
                            return std::move(std::get<value_type>(v));
                        }
                    }
                    else if (std::holds_alternative<error_type>(v))
                    {
                        std::visit(
                            sync_wait_error_visitor{}, std::get<error_type>(v));
                    }

                    // If the variant holds a std::monostate something has gone
                    // wrong and we terminate
                    HPX_UNREACHABLE;
                }
            };

            state& st;

            void signal_set_called() noexcept
            {
                std::unique_lock<hpx::lcos::local::mutex> l(st.m);
                st.set_called = true;
                st.cv.notify_one();
            }

            template <typename E>
                void set_error(E&& e) && noexcept
            {
                st.v.template emplace<error_type>(std::forward<E>(e));
                signal_set_called();
            }

            void set_done() && noexcept
            {
                signal_set_called();
            };

            void set_value() && noexcept
            {
                st.v.template emplace<value_type>();
                signal_set_called();
            }

            template <typename U>
                void set_value(U&& u) && noexcept
            {
                st.v.template emplace<value_type>(std::forward<U>(u));
                signal_set_called();
            }
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct sync_wait_t final
      : hpx::functional::tag_fallback<sync_wait_t>
    {
    private:
        template <typename S>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            sync_wait_t, S&& s)
        {
            using receiver_type = detail::sync_wait_receiver<S>;
            using state_type = typename receiver_type::state;

            state_type st{};
            auto os = hpx::execution::experimental::connect(
                std::forward<S>(s), receiver_type{st});
            hpx::execution::experimental::start(os);

            st.wait();
            return st.get_value();
        }

        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(sync_wait_t)
        {
            return detail::partial_algorithm<sync_wait_t>{};
        }
    } sync_wait{};
}}}    // namespace hpx::execution::experimental
#endif
