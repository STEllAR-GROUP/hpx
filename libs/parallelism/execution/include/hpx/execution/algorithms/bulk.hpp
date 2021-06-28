//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/transform.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/iterator_support/counting_iterator.hpp>
#include <hpx/iterator_support/iterator_range.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/type_support/pack.hpp>

#include <exception>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace execution { namespace experimental {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename R, typename Shape, typename F>
        struct bulk_receiver
        {
            std::decay_t<R> r;
            std::decay_t<Shape> shape;
            std::decay_t<F> f;

            using shape_iterator = typename hpx::traits::range_traits<
                std::decay_t<Shape>>::iterator_type;
            using shape_element =
                typename std::iterator_traits<shape_iterator>::value_type;

            template <typename R_, typename Shape_, typename F_>
            bulk_receiver(R_&& r, Shape_&& shape, F_&& f)
              : r(std::forward<R_>(r))
              , shape(std::forward<Shape_>(shape))
              , f(std::forward<F_>(f))
            {
            }

            template <typename E>
            void set_error(E&& e) && noexcept
            {
                hpx::execution::experimental::set_error(
                    std::move(r), std::forward<E>(e));
            }

            void set_done() && noexcept
            {
                hpx::execution::experimental::set_done(std::move(r));
            }

        private:
            template <typename... Ts>
            void set_value_helper(std::true_type, Ts&&... ts) noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        for (auto const& s : shape)
                        {
                            HPX_INVOKE(f, s, ts...);
                        }
                        hpx::execution::experimental::set_value(std::move(r));
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            std::move(r), std::move(ep));
                    });
            }

            template <typename... Ts>
            void set_value_helper(std::false_type, Ts&&... ts) noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        using result_type =
                            hpx::util::invoke_result_t<std::decay_t<F>,
                                shape_element, std::decay_t<Ts>...>;

                        std::vector<result_type> results;
                        results.reserve(util::size(shape));
                        for (auto const& s : shape)
                        {
                            results.emplace_back(HPX_INVOKE(f, s, ts...));
                        }

                        hpx::execution::experimental::set_value(
                            std::move(r), std::move(results));
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            std::move(r), std::move(ep));
                    });
            }

        public:
            template <typename... Ts,
                typename = std::enable_if_t<
                    hpx::is_invocable_v<F, shape_element, Ts...>>>
            void set_value(Ts&&... ts) && noexcept
            {
                using is_void_result = std::is_void<
                    hpx::util::invoke_result_t<F, shape_element, Ts...>>;
                set_value_helper(is_void_result{}, std::forward<Ts>(ts)...);
            }
        };

        template <typename S, typename Shape, typename F>
        struct bulk_sender
        {
            std::decay_t<S> s;
            std::decay_t<Shape> shape;
            std::decay_t<F> f;

            using shape_iterator = typename hpx::traits::range_traits<
                std::decay_t<Shape>>::iterator_type;
            using shape_element =
                typename std::iterator_traits<shape_iterator>::value_type;

            template <typename Tuple>
            struct invoke_result_helper;

            template <template <typename...> class Tuple, typename... Ts>
            struct invoke_result_helper<Tuple<Ts...>>
            {
                using result_type =
                    hpx::util::invoke_result_t<F, shape_element, Ts...>;
                using type =
                    typename std::conditional<std::is_void<result_type>::value,
                        Tuple<>, Tuple<std::vector<result_type>>>::type;
            };

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types =
                hpx::util::detail::unique_t<hpx::util::detail::transform_t<
                    typename hpx::execution::experimental::sender_traits<
                        S>::template value_types<Tuple, Variant>,
                    invoke_result_helper>>;

            template <template <typename...> class Variant>
            using error_types =
                hpx::util::detail::unique_t<hpx::util::detail::prepend_t<
                    typename hpx::execution::experimental::sender_traits<
                        S>::template error_types<Variant>,
                    std::exception_ptr>>;

            static constexpr bool sends_done = false;

            template <typename R>
            auto connect(R&& r) &&
            {
                return hpx::execution::experimental::connect(std::move(s),
                    bulk_receiver<R, Shape, F>(
                        std::forward<R>(r), std::move(shape), std::move(f)));
            }

            template <typename R>
            auto connect(R&& r) &
            {
                return hpx::execution::experimental::connect(s,
                    bulk_receiver<R, Shape, F>(std::forward<R>(r), shape, f));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Incrementable>
        using counting_shape_type = hpx::util::iterator_range<
            hpx::util::counting_iterator<Incrementable>>;

        template <typename Incrementable>
        HPX_HOST_DEVICE inline counting_shape_type<Incrementable>
        make_counting_shape(Incrementable n)
        {
            return hpx::util::make_iterator_range(
                hpx::util::make_counting_iterator(Incrementable(0)),
                hpx::util::make_counting_iterator(n));
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    HPX_INLINE_CONSTEXPR_VARIABLE struct bulk_t final
      : hpx::functional::tag_fallback<bulk_t>
    {
    private:
        // clang-format off
        template <typename S, typename Shape, typename F,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<S> &&
                std::is_integral<Shape>::value
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            bulk_t, S&& s, Shape const& shape, F&& f)
        {
            return detail::bulk_sender<S, detail::counting_shape_type<Shape>,
                F>{std::forward<S>(s), detail::make_counting_shape(shape),
                std::forward<F>(f)};
        }

        // clang-format off
        template <typename S, typename Shape, typename F,
            HPX_CONCEPT_REQUIRES_(
                is_sender_v<S> &&
                !std::is_integral<std::decay_t<Shape>>::value
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            bulk_t, S&& s, Shape&& shape, F&& f)
        {
            return detail::bulk_sender<S, Shape, F>{std::forward<S>(s),
                std::forward<Shape>(shape), std::forward<F>(f)};
        }

        // clang-format off
        template <typename Shape, typename F,
            HPX_CONCEPT_REQUIRES_(
                std::is_integral<Shape>::value
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            bulk_t, Shape const& shape, F&& f)
        {
            return detail::partial_algorithm<bulk_t,
                detail::counting_shape_type<Shape>, F>{
                detail::make_counting_shape(shape), std::forward<F>(f)};
        }

        // clang-format off
        template <typename Shape, typename F,
            HPX_CONCEPT_REQUIRES_(
                !std::is_integral<std::decay_t<Shape>>::value
            )>
        // clang-format on
        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(
            bulk_t, Shape&& shape, F&& f)
        {
            return detail::partial_algorithm<bulk_t, Shape, F>{
                std::forward<Shape>(shape), std::forward<F>(f)};
        }
    } bulk{};
}}}    // namespace hpx::execution::experimental
