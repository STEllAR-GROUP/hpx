//  Copyright (c) 2017 Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_PACK_TRAVERSAL_ASYNC_HPP
#define HPX_UTIL_PACK_TRAVERSAL_ASYNC_HPP

#include <hpx/util/detail/pack_traversal_async_impl.hpp>

#include <utility>

namespace hpx {
namespace util {
    /// Traverses the pack with the given visitor in an asynchronous way.
    ///
    /// This function works in the same way as `traverse_pack`,
    /// however, we are able to suspend and continue the traversal at
    /// later time.
    /// Thus we require a visitor callable object which provides three
    /// `operator()` overloads as depicted by the code sample below:
    ///    ```cpp
    ///    struct my_async_visitor
    ///    {
    ///        /// The synchronous overload is called for each object,
    ///        /// it may return false to suspend the current control.
    ///        /// In that case the overload below is called.
    ///        template <typename T>
    ///        bool operator()(T&& element)
    ///        {
    ///            return true;
    ///        }
    ///
    ///        /// The asynchronous overload this is called when the
    ///        /// synchronous overload returned false.
    ///        /// In addition to the current visited element the overload is
    ///        /// called with a contnuation callable object which resumes the
    ///        /// traversal when it's called later.
    ///        /// The continuation next may be stored and called later or
    ///        /// dropped completely to abort the traversal early.
    ///        template <typename T, typename N>
    ///        void operator()(T&& element, N&& next)
    ///        {
    ///        }
    ///
    ///        /// The overload with no parameters is called when the traversal
    ///        /// was finished.
    ///        void operator()()
    ///        {
    ///        }
    ///    };
    ///    ```
    ///
    /// See `traverse_pack` for a detailed description.
    template <typename Visitor, typename... T>
    void traverse_pack_async(Visitor&& visitor, T&&... pack)
    {
        detail::apply_pack_transform_async(
            std::forward<Visitor>(visitor), std::forward<T>(pack)...);
    }
}    // end namespace util
}    // end namespace hpx

#endif    // HPX_UTIL_PACK_TRAVERSAL_ASYNC_HPP
