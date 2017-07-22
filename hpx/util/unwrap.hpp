//  Copyright (c) 2017 Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_UNWRAP_HPP
#define HPX_UTIL_UNWRAP_HPP

#include <hpx/config.hpp>
#include <hpx/util/detail/unwrap_impl.hpp>

#include <cstddef>
#include <utility>

namespace hpx {
namespace util {
    /// A helper function for retrieving the actual result of
    /// any hpx::lcos::future like type which is wrapped in an arbitrary way.
    ///
    /// Unwraps the given pack of arguments, so that any hpx::lcos::future
    /// object is replaced by its future result type in the argument pack:
    /// - `hpx::future<int>` -> `int`
    /// - `hpx::future<std::vector<float>>` -> `std::vector<float>`
    /// - `std::vector<future<float>>` -> `std::vector<float>`
    ///
    /// The function is capable of unwrapping hpx::lcos::future like objects
    /// that are wrapped inside any container or tuple like type,
    /// see hpx::util::map_pack() for a detailed description about which
    /// surrounding types are supported.
    /// Non hpx::lcos::future like types are permitted as arguments and
    /// passed through.
    ///
    ///   ```cpp
    ///   // Single arguments
    ///   int i1 = hpx:util::unwrap(hpx::lcos::make_ready_future(0));
    ///
    ///   // Multiple arguments
    ///   hpx::tuple<int, int> i2 =
    ///       hpx:util::unwrap(hpx::lcos::make_ready_future(1),
    ///                        hpx::lcos::make_ready_future(2));
    ///   ```
    ///
    /// \note    This function unwraps the given arguments until the first
    ///          traversed nested hpx::lcos::future which corresponds to
    ///          an unwrapping depth of one.
    ///          See hpx::util::unwrap_n() for a function which unwraps the
    ///          given arguments to a particular depth or
    ///          hpx::util::unwrap_all() that unwraps all future like objects
    ///          recursively which are contained in the arguments.
    ///
    /// \param   args the arguments that are unwrapped which may contain any
    ///          arbitrary future or non future type.
    ///
    /// \returns Depending on the count of arguments this function returns
    ///          a hpx::util::tuple containing the unwrapped arguments
    ///          if multiple arguments are given.
    ///          In case the function is called with a single argument,
    ///          the argument is unwrapped and returned.
    ///
    /// \throws  std::exception like objects in case any of the given wrapped
    ///          hpx::lcos::future objects were resolved through an exception.
    ///          See hpx::lcos::future::get() for details.
    ///
    template <typename... Args>
    auto unwrap(Args&&... args)
        -> decltype(detail::unwrap_depth_impl<1U>(std::forward<Args>(args)...))
    {
        return detail::unwrap_depth_impl<1U>(std::forward<Args>(args)...);
    }

    /// An alterntive version of hpx::util::unwrap(), which unwraps the given
    /// arguments to a certain depth of hpx::lcos::future like objects.
    ///
    /// \tparam Depth The count of hpx::lcos::future like objects which are
    ///               unwrapped maximally.
    ///
    /// See unwrap for a detailed description.
    ///
    template <std::size_t Depth, typename... Args>
    auto unwrap_n(Args&&... args) -> decltype(
        detail::unwrap_depth_impl<Depth>(std::forward<Args>(args)...))
    {
        static_assert(Depth > 0U, "The unwrapping depth must be >= 1!");
        return detail::unwrap_depth_impl<Depth>(std::forward<Args>(args)...);
    }

    /// An alterntive version of hpx::util::unwrap(), which unwraps the given
    /// arguments recursively so that all contained hpx::lcos::future like
    /// objects are replaced by their actual value.
    ///
    /// See hpx::util::unwrap() for a detailed description.
    ///
    template <typename... Args>
    auto unwrap_all(Args&&... args)
        -> decltype(detail::unwrap_depth_impl<0U>(std::forward<Args>(args)...))
    {
        return detail::unwrap_depth_impl<0U>(std::forward<Args>(args)...);
    }

    /// Returns a callable object which unwraps its arguments upon
    /// invocation using the hpx::util::unwrap() function and then passes
    /// the result to the given callable object.
    ///
    ///   ```cpp
    ///   auto callable = hpx::util::unwrapping([](int left, int right) {
    ///       return left + right;
    ///   });
    ///
    ///   int i1 = callable(hpx::lcos::make_ready_future(1),
    ///                     hpx::lcos::make_ready_future(2));
    ///   ```
    ///
    /// See hpx::util::unwrap() for a detailed description.
    ///
    /// \param callable the callable object which which is called with
    ///        the result of the corresponding unwrap function.
    ///
    template <typename T>
    auto unwrapping(T&& callable) -> decltype(
        detail::functional_unwrap_depth_impl<1U>(std::forward<T>(callable)))
    {
        return detail::functional_unwrap_depth_impl<1U>(
            std::forward<T>(callable));
    }

    /// Returns a callable object which unwraps its arguments upon
    /// invocation using the hpx::util::unwrap_n() function and then passes
    /// the result to the given callable object.
    ///
    /// See hpx::util::unwrapping() for a detailed description.
    ///
    template <std::size_t Depth, typename T>
    auto unwrapping_n(T&& callable) -> decltype(
        detail::functional_unwrap_depth_impl<Depth>(std::forward<T>(callable)))
    {
        static_assert(Depth > 0U, "The unwrapping depth must be >= 1!");
        return detail::functional_unwrap_depth_impl<Depth>(
            std::forward<T>(callable));
    }

    /// Returns a callable object which unwraps its arguments upon
    /// invocation using the hpx::util::unwrap_all() function and then passes
    /// the result to the given callable object.
    ///
    /// See hpx::util::unwrapping() for a detailed description.
    ///
    template <typename T>
    auto unwrapping_all(T&& callable) -> decltype(
        detail::functional_unwrap_depth_impl<0U>(std::forward<T>(callable)))
    {
        return detail::functional_unwrap_depth_impl<0U>(
            std::forward<T>(callable));
    }
}    // end namespace util
}    // end namespace hpx

#endif    // HPX_UTIL_UNWRAP_HPP
