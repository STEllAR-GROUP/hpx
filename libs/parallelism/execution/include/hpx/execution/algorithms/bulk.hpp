//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/datastructures/variant.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/transform.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/iterator_support/counting_shape.hpp>
#include <hpx/type_support/pack.hpp>

#include <exception>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution { namespace experimental {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename Sender, typename Shape, typename F>
        struct bulk_sender
        {
            std::decay_t<Sender> sender;
            std::decay_t<Shape> shape;
            std::decay_t<F> f;

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types =
                typename hpx::execution::experimental::sender_traits<
                    Sender>::template value_types<Tuple, Variant>;

            template <template <typename...> class Variant>
            using error_types =
                hpx::util::detail::unique_t<hpx::util::detail::prepend_t<
                    typename hpx::execution::experimental::sender_traits<
                        Sender>::template error_types<Variant>,
                    std::exception_ptr>>;

            static constexpr bool sends_done = false;

            template <typename Receiver>
            auto connect(Receiver&& receiver) &&
            {
                return hpx::execution::experimental::connect(std::move(sender),
                    bulk_receiver<Receiver, Shape, F>(
                        std::forward<Receiver>(receiver), std::move(shape),
                        std::move(f)));
            }

            template <typename Receiver>
            auto connect(Receiver&& receiver) &
            {
                return hpx::execution::experimental::connect(sender,
                    bulk_receiver<Receiver, Shape, F>(
                        std::forward<Receiver>(receiver), shape, f));
            }

            template <typename Receiver, typename Shape_, typename F_>
            struct bulk_receiver
            {
                std::decay_t<Receiver> receiver;
                std::decay_t<Shape_> shape;
                std::decay_t<F_> f;

                template <typename Receiver_, typename Shape__, typename F__>
                bulk_receiver(Receiver_&& receiver, Shape__&& shape, F__&& f)
                  : receiver(std::forward<Receiver_>(receiver))
                  , shape(std::forward<Shape__>(shape))
                  , f(std::forward<F__>(f))
                {
                }

                template <typename E>
                void set_error(E&& e) && noexcept
                {
                    hpx::execution::experimental::set_error(
                        std::move(receiver), std::forward<E>(e));
                }

                void set_done() && noexcept
                {
                    hpx::execution::experimental::set_done(std::move(receiver));
                }

                // The typedef is duplicated from parent struct as the parent one is
                // not instantiated early enough to use it here.
                using value_type =
                    typename hpx::execution::experimental::sender_traits<
                        Sender>::template value_types<hpx::tuple, hpx::variant>;

                template <typename... Ts>
                auto set_value(Ts&&... ts) && noexcept -> decltype(
                    std::declval<hpx::variant<hpx::monostate, value_type>>()
                        .template emplace<value_type>(
                            hpx::make_tuple<>(std::forward<Ts>(ts)...)),
                    void())
                {
                    hpx::detail::try_catch_exception_ptr(
                        [&]() {
                            for (auto const& s : shape)
                            {
                                HPX_INVOKE(f, s, ts...);
                            }
                            hpx::execution::experimental::set_value(
                                std::move(receiver), std::forward<Ts>(ts)...);
                        },
                        [&](std::exception_ptr ep) {
                            hpx::execution::experimental::set_error(
                                std::move(receiver), std::move(ep));
                        });
                }
            };
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    HPX_INLINE_CONSTEXPR_VARIABLE struct bulk_t final
      : hpx::functional::tag_fallback<bulk_t>
    {
    private:
        // clang-format off
        template <typename Sender, typename Shape, typename F,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                std::is_integral<Shape>::value
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            bulk_t, Sender&& sender, Shape const& shape, F&& f)
        {
            return detail::bulk_sender<Sender,
                hpx::util::counting_shape_type<Shape>, F>{
                std::forward<Sender>(sender),
                hpx::util::make_counting_shape(shape), std::forward<F>(f)};
        }

        // clang-format off
        template <typename Sender, typename Shape, typename F,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                !std::is_integral<std::decay_t<Shape>>::value
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            bulk_t, Sender&& sender, Shape&& shape, F&& f)
        {
            return detail::bulk_sender<Sender, Shape, F>{
                std::forward<Sender>(sender), std::forward<Shape>(shape),
                std::forward<F>(f)};
        }

        template <typename Shape, typename F>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            bulk_t, Shape&& shape, F&& f)
        {
            return detail::partial_algorithm<bulk_t, Shape, F>{
                std::forward<Shape>(shape), std::forward<F>(f)};
        }
    } bulk{};
}}}    // namespace hpx::execution::experimental
