//  Copyright (c) 2017 Denis Blank
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/pack_traversal/detail/unwrap_impl.hpp>

#include <cstddef>
#include <utility>

namespace hpx {

    /// A helper function for retrieving the actual result of any hpx::future
    /// like type which is wrapped in an arbitrary way.
    ///
    /// Unwraps the given pack of arguments, so that any hpx::future object is
    /// replaced by its future result type in the argument pack:
    /// - `hpx::future<int>` -> `int`
    /// - `hpx::future<std::vector<float>>` -> `std::vector<float>`
    /// - `std::vector<future<float>>` -> `std::vector<float>`
    ///
    /// The function is capable of unwrapping hpx::future like objects that are
    /// wrapped inside any container or tuple like type, see
    /// hpx::util::map_pack() for a detailed description about which surrounding
    /// types are supported. Non hpx::future like types are permitted as
    /// arguments and passed through.
    ///
    ///   ```cpp // Single arguments int i1 =
    ///   hpx:unwrap(hpx::make_ready_future(0));
    ///
    ///   // Multiple arguments hpx::tuple<int, int> i2 =
    ///       hpx:unwrap(hpx::make_ready_future(1),
    ///                  hpx::make_ready_future(2));
    ///   ```
    ///
    /// \note    This function unwraps the given arguments until the first
    ///          traversed nested hpx::future which corresponds to an unwrapping
    ///          depth of one. See hpx::unwrap_n() for a function which unwraps
    ///          the given arguments to a particular depth or hpx::unwrap_all()
    ///          that unwraps all future like objects recursively which are
    ///          contained in the arguments.
    ///
    /// \param   args the arguments that are unwrapped which may contain any
    ///          arbitrary future or non future type.
    ///
    /// \returns Depending on the count of arguments this function returns a
    ///          hpx::tuple containing the unwrapped arguments if multiple
    ///          arguments are given. In case the function is called with a
    ///          single argument, the argument is unwrapped and returned.
    ///
    /// \throws  std::exception like objects in case any of the given wrapped
    ///          hpx::future objects were resolved through an exception. See
    ///          hpx::future::get() for details.
    ///
    template <typename... Args>
    auto unwrap(Args&&... args) -> decltype(
        util::detail::unwrap_depth_impl<1U>(HPX_FORWARD(Args, args)...))
    {
        return util::detail::unwrap_depth_impl<1U>(HPX_FORWARD(Args, args)...);
    }

    namespace functional {

        /// A helper function object for functionally invoking `hpx::unwrap`.
        /// For more information please refer to its documentation.
        struct unwrap
        {
            /// \cond NOINTERNAL
            template <typename... Args>
            auto operator()(Args&&... args)
                -> decltype(hpx::unwrap(HPX_FORWARD(Args, args)...))
            {
                return hpx::unwrap(HPX_FORWARD(Args, args)...);
            }
            /// \endcond
        };
    }    // namespace functional

    /// An alterntive version of hpx::unwrap(), which unwraps the given
    /// arguments to a certain depth of hpx::future like objects.
    ///
    /// \tparam Depth The count of hpx::future like objects which are
    ///               unwrapped maximally.
    ///
    /// See unwrap for a detailed description.
    ///
    template <std::size_t Depth, typename... Args>
    auto unwrap_n(Args&&... args) -> decltype(
        util::detail::unwrap_depth_impl<Depth>(HPX_FORWARD(Args, args)...))
    {
        static_assert(Depth > 0U, "The unwrapping depth must be >= 1!");
        return util::detail::unwrap_depth_impl<Depth>(
            HPX_FORWARD(Args, args)...);
    }

    namespace functional {

        /// A helper function object for functionally invoking `hpx::unwrap_n`.
        /// For more information please refer to its documentation.
        template <std::size_t Depth>
        struct unwrap_n
        {
            /// \cond NOINTERNAL
            template <typename... Args>
            auto operator()(Args&&... args)
                -> decltype(hpx::unwrap_n<Depth>(HPX_FORWARD(Args, args)...))
            {
                return hpx::unwrap_n<Depth>(HPX_FORWARD(Args, args)...);
            }
            /// \endcond
        };
    }    // namespace functional

    /// An alterntive version of hpx::unwrap(), which unwraps the given
    /// arguments recursively so that all contained hpx::future like objects are
    /// replaced by their actual value.
    ///
    /// See hpx::unwrap() for a detailed description.
    ///
    template <typename... Args>
    auto unwrap_all(Args&&... args) -> decltype(
        util::detail::unwrap_depth_impl<0U>(HPX_FORWARD(Args, args)...))
    {
        return util::detail::unwrap_depth_impl<0U>(HPX_FORWARD(Args, args)...);
    }

    namespace functional {

        /// A helper function object for functionally invoking
        /// `hpx::unwrap_all`. For more information please refer to its
        /// documentation.
        struct unwrap_all
        {
            /// \cond NOINTERNAL
            template <typename... Args>
            auto operator()(Args&&... args)
                -> decltype(hpx::unwrap_all(HPX_FORWARD(Args, args)...))
            {
                return hpx::unwrap_all(HPX_FORWARD(Args, args)...);
            }
            /// \endcond
        };
    }    // namespace functional

    /// Returns a callable object which unwraps its arguments upon invocation
    /// using the hpx::unwrap() function and then passes the result to the given
    /// callable object.
    ///
    ///   ```cpp auto callable = hpx::unwrapping([](int left, int right) {
    ///       return left + right;
    ///   });
    ///
    ///   int i1 = callable(hpx::make_ready_future(1),
    ///                     hpx::make_ready_future(2));
    ///   ```
    ///
    /// See hpx::unwrap() for a detailed description.
    ///
    /// \param callable the callable object which which is called with the
    ///        result of the corresponding unwrap function.
    ///
    template <typename T>
    auto unwrapping(T&& callable)
        -> decltype(util::detail::functional_unwrap_depth_impl<1U>(
            HPX_FORWARD(T, callable)))
    {
        return util::detail::functional_unwrap_depth_impl<1U>(
            HPX_FORWARD(T, callable));
    }

    /// Returns a callable object which unwraps its arguments upon invocation
    /// using the hpx::unwrap_n() function and then passes the result to the
    /// given callable object.
    ///
    /// See hpx::unwrapping() for a detailed description.
    ///
    template <std::size_t Depth, typename T>
    auto unwrapping_n(T&& callable)
        -> decltype(util::detail::functional_unwrap_depth_impl<Depth>(
            HPX_FORWARD(T, callable)))
    {
        static_assert(Depth > 0U, "The unwrapping depth must be >= 1!");
        return util::detail::functional_unwrap_depth_impl<Depth>(
            HPX_FORWARD(T, callable));
    }

    /// Returns a callable object which unwraps its arguments upon invocation
    /// using the hpx::unwrap_all() function and then passes the result to the
    /// given callable object.
    ///
    /// See hpx::unwrapping() for a detailed description.
    ///
    template <typename T>
    auto unwrapping_all(T&& callable)
        -> decltype(util::detail::functional_unwrap_depth_impl<0U>(
            HPX_FORWARD(T, callable)))
    {
        return util::detail::functional_unwrap_depth_impl<0U>(
            HPX_FORWARD(T, callable));
    }
}    // namespace hpx
