//  Copyright (c)      2013 Thomas Heller
//  Copyright (c)      2014 Agustin Berge
//  Copyright (c)      2017 Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_UNWRAPPED_HPP
#define HPX_UTIL_UNWRAPPED_HPP

#include <hpx/config.hpp>

// hpx::util::unwrapped was deprecated in V1.1.0 and thus will be
// removed in a later version of HPX.
#if defined(HPX_HAVE_UNWRAPPED_COMPATIBILITY)

#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_future_range.hpp>
#include <hpx/traits/is_future_tuple.hpp>
#include <hpx/util/unwrap.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx {
namespace util {
    namespace detail {
        /// Defines a trait which deduces to a std::true_type
        /// if the functional version of unwrapped should be used.
        /// This trait defines the original behaviour of unwrapped and
        /// shoudldn't be used except to proxy the old unwrapped function
        /// to its new version unwrap.
        ///
        /// This requires two satisfied conditions which arise from the
        /// implementation of the old and deprecated unwrapped implementation:
        /// - There is one argument in the variadic argument pack.
        /// - The only argument is a future itself (traits::is_future)
        ///   or a range of futures (traits::is_future_range)
        ///   or a tuple of futures (traits::is_future_range).
        template <typename... Args>
        struct is_functional_unwrap;
        template <typename First, typename Second, typename... Args>
        struct is_functional_unwrap<First, Second, Args...> : std::false_type
        {
        };
        template <typename First>
        struct is_functional_unwrap<First>
          : std::integral_constant<bool,
                !(traits::is_future<typename decay<First>::type>::value ||
                    traits::is_future_range<
                        typename decay<First>::type>::value ||
                    traits::is_future_tuple<
                        typename decay<First>::type>::value)>
        {
        };

        /// Proxy class for choosing for forwarding the input to the
        /// corresponding rewritten version of unwrapped (unwrap).
        template <bool IsFunctional /*= false*/, std::size_t Depth>
        struct proxy_new_unwrapped
        {
            /// The immediate unwrap
            template <typename... Args>
            static auto proxy(Args&&... args)
                -> decltype(unwrap_n<Depth>(std::forward<Args>(args)...))
            {
                return unwrap_n<Depth>(std::forward<Args>(args)...);
            }
        };
        template <std::size_t Depth>
        struct proxy_new_unwrapped<true, Depth>
        {
            /// The functional unwrap
            template <typename Callable>
            static auto proxy(Callable&& callable) -> decltype(
                unwrapping_n<Depth>(std::forward<Callable>(callable)))
            {
                return unwrapping_n<Depth>(std::forward<Callable>(callable));
            }
        };
    }    // end namespace detail

    /// A multi-usable helper function for retrieving the actual result of
    /// any hpx::lcos::future which is wrapped in an arbitrary way.
    ///
    /// unwrapped supports multiple applications, the underlying
    /// implementation is chosen based on the given arguments:
    ///
    /// - For a single callable object as argument,
    ///   the **deferred form** is used, which makes the function to return a
    ///   callable object that unwraps the input and passes it to the
    ///   given callable object upon invocation:
    ///   ```cpp
    ///   auto add = [](int left, int right) {
    ///       return left + right;
    ///   };
    ///   auto unwrapper = hpx:util:::unwrapped(add);
    ///   hpx::util::tuple<hpx::future<int>, hpx::future<int>> tuple = ...;
    ///   int result = unwrapper(tuple);
    ///   ```
    ///   The `unwrapper` object could be used to connect the `add` function
    ///   to the continuation handler of a hpx::future.
    ///
    /// - For any other input, the **immediate form** is used,
    ///   which unwraps the given pack of arguments,
    ///   so that any hpx::lcos::future object is replaced by
    ///   its future result type in the pack:
    ///       - `hpx::future<int>` -> `int`
    ///       - `hpx::future<std::vector<float>>` -> `std::vector<float>`
    ///       - `std::vector<future<float>>` -> `std::vector<float>`
    ///
    /// \param   args the argument pack that determines the used implementation
    ///
    /// \returns Depending on the chosen implementation the return type is
    ///          either a hpx::util::tuple containing unwrapped hpx::futures
    ///          when the *immediate form* is used.
    ///          If the *deferred form* is used, the function returns a
    ///          callable object, which unwrapps and forwards its arguments
    ///          when called, as desribed above.
    ///
    /// \throws std::exception like object in case the immediate application is
    ///         used and if any of the given hpx::lcos::future objects were
    ///         resolved through an exception.
    ///         See hpx::lcos::future::get() for details.
    ///
    /// \deprecated hpx::util::unwrapped was replaced by hpx::util::unwrap
    ///             and hpx::util::unwrapping and might be removed in a
    ///             later version of HPX!
    ///             The main reason for deprecation was that the automatic
    ///             callable type detection doesn't anymore correctly,
    ///             as soon as we allowed to route non future types through.
    ///
    template <typename... Args>
    HPX_DEPRECATED("hpx::util::unwrapped was replaced by "
                   "hpx::util::unwrap and hpx::util::unwrapping "
                   "and might be removed in a later version of HPX!")
    auto unwrapped(Args&&... args) -> decltype(detail::proxy_new_unwrapped<
        detail::is_functional_unwrap<Args...>::value,
        1U>::proxy(std::forward<Args>(args)...))
    {
        return detail::proxy_new_unwrapped<
            detail::is_functional_unwrap<Args...>::value,
            1U>::proxy(std::forward<Args>(args)...);
    }

    /// Provides an additional implementation of unwrapped which
    /// unwraps nested hpx::futures within a two-level depth.
    ///
    /// \deprecated hpx::util::unwrapped2 was replaced by hpx::util::unwrap_n<2>
    ///             and hpx::util::unwrapping_n<2> and might be removed in a
    ///             later version of HPX!
    ///
    /// See hpx::util::unwrapped() for details.
    ///
    template <typename... Args>
    HPX_DEPRECATED("hpx::util::unwrapped2 was replaced by "
                   "hpx::util::unwrap_n<2> and hpx::util::unwrapping_n<2> "
                   "and might be removed in a later version of HPX!")
    auto unwrapped2(Args&&... args) -> decltype(detail::proxy_new_unwrapped<
        detail::is_functional_unwrap<Args...>::value,
        2U>::proxy(std::forward<Args>(args)...))
    {
        return detail::proxy_new_unwrapped<
            detail::is_functional_unwrap<Args...>::value,
            2U>::proxy(std::forward<Args>(args)...);
    }
}    // end namespace util
}    // end namespace hpx

#endif    // HPX_HAVE_UNWRAPPED_COMPATIBILITY
#endif    // HPX_UTIL_UNWRAPPED_HPP
