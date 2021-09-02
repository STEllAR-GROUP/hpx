//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/datastructures/variant.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/transform.hpp>
#include <hpx/execution_base/completion_scheduler.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/tag_priority_dispatch.hpp>
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
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Sender> sender;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Shape> shape;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;

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

            template <typename CPO,
                // clang-format off
                HPX_CONCEPT_REQUIRES_(
                    hpx::execution::experimental::detail::is_receiver_cpo_v<CPO> &&
                    hpx::execution::experimental::detail::has_completion_scheduler_v<
                        CPO, std::decay_t<Sender>>)
                // clang-format on
                >
            friend constexpr auto tag_dispatch(
                hpx::execution::experimental::get_completion_scheduler_t<CPO>,
                bulk_sender const& sender)
            {
                return hpx::execution::experimental::get_completion_scheduler<
                    CPO>(sender.sender);
            }

            template <typename Receiver>
            struct bulk_receiver
            {
                HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;
                HPX_NO_UNIQUE_ADDRESS std::decay_t<Shape> shape;
                HPX_NO_UNIQUE_ADDRESS std::decay_t<F> f;

                template <typename Receiver_, typename Shape_, typename F_>
                bulk_receiver(Receiver_&& receiver, Shape_&& shape, F_&& f)
                  : receiver(std::forward<Receiver_>(receiver))
                  , shape(std::forward<Shape_>(shape))
                  , f(std::forward<F_>(f))
                {
                }

                template <typename Error>
                friend void tag_dispatch(
                    set_error_t, bulk_receiver&& r, Error&& error) noexcept
                {
                    hpx::execution::experimental::set_error(
                        std::move(r.receiver), std::forward<Error>(error));
                }

                friend void tag_dispatch(set_done_t, bulk_receiver&& r) noexcept
                {
                    hpx::execution::experimental::set_done(
                        std::move(r.receiver));
                }

                template <typename... Ts>
                void set_value(Ts&&... ts)
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

                template <typename... Ts>
                friend auto tag_dispatch(
                    set_value_t, bulk_receiver&& r, Ts&&... ts) noexcept
                    -> decltype(hpx::execution::experimental::set_value(
                                    std::declval<std::decay_t<Receiver>&&>(),
                                    std::forward<Ts>(ts)...),
                        void())
                {
                    // set_value is in a member function only because of a
                    // compiler bug in GCC 7. When the body of set_value is
                    // inlined here compilation fails with an internal compiler
                    // error.
                    r.set_value(std::forward<Ts>(ts)...);
                }
            };

            template <typename Receiver>
            friend auto tag_dispatch(
                connect_t, bulk_sender&& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(
                    std::move(s.sender),
                    bulk_receiver<Receiver>(std::forward<Receiver>(receiver),
                        std::move(s.shape), std::move(s.f)));
            }

            template <typename Receiver>
            friend auto tag_dispatch(
                connect_t, bulk_sender& s, Receiver&& receiver)
            {
                return hpx::execution::experimental::connect(s.sender,
                    bulk_receiver<Receiver>(
                        std::forward<Receiver>(receiver), s.shape, s.f));
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    HPX_INLINE_CONSTEXPR_VARIABLE struct bulk_t final
      : hpx::functional::tag_priority<bulk_t>
    {
    private:
        // clang-format off
        template <typename Sender, typename Shape, typename F,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<Sender> &&
                hpx::execution::experimental::detail::
                    is_completion_scheduler_tag_dispatchable_v<
                        hpx::execution::experimental::set_value_t, Sender,
                        bulk_t, Shape, F>)>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_override_dispatch(
            bulk_t, Sender&& sender, Shape const& shape, F&& f)
        {
            auto scheduler =
                hpx::execution::experimental::get_completion_scheduler<
                    hpx::execution::experimental::set_value_t>(sender);
            return hpx::functional::tag_dispatch(bulk_t{}, std::move(scheduler),
                std::forward<Sender>(sender), shape, std::forward<F>(f));
        }

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
                hpx::util::detail::counting_shape_type<Shape>, F>{
                std::forward<Sender>(sender),
                hpx::util::detail::make_counting_shape(shape),
                std::forward<F>(f)};
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
