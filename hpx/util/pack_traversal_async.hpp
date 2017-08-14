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
    /// A tag which is passed to the `operator()` of the visitor
    /// if an element is visited synchronously.
    using detail::async_traverse_visit_tag;
    /// A tag which is passed to the `operator()` of the visitor
    /// if an element is visited after the traversal was detached.
    using detail::async_traverse_detach_tag;
    /// A tag which is passed to the `operator()` of the visitor
    /// if the asynchronous pack traversal was finished.
    using detail::async_traverse_complete_tag;

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
    ///        bool operator()(async_traverse_visit_tag, T&& element)
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
    ///        void operator()(async_traverse_detach_tag, T&& element, N&& next)
    ///        {
    ///        }
    ///
    ///        /// The overload is called when the traversal was finished.
    ///        /// As argument the whole pack is passed over which we
    ///        /// traversed asynchrnously.
    ///        template <typename T>
    ///        void operator()(async_traverse_complete_tag, T&& pack)
    ///        {
    ///        }
    ///    };
    ///    ```
    ///
    /// \param   visitor A visitor object which provides the three `operator()`
    ///                  overloads that were described above.
    ///                  Additionally the visitor must be compatible
    ///                  for referencing it from a `boost::intrusive_ptr`.
    ///
    /// \param   pack    The arbitrary parameter pack which is traversed
    ///                  asynchronously. Nested objects inside containers and
    ///                  tuple like types are traversed recursively.
    ///
    /// \returns         A boost::intrusive_ptr that references an instance of
    ///                  the given visitor object.
    ///
    /// See `traverse_pack` for a detailed description about the
    /// traversal behaviour and capabilities.
    ///
    template <typename Visitor, typename... T>
    auto traverse_pack_async(Visitor&& visitor, T&&... pack)
        -> decltype(detail::apply_pack_transform_async(
            std::forward<Visitor>(visitor), std::forward<T>(pack)...))
    {
        return detail::apply_pack_transform_async(
            std::forward<Visitor>(visitor), std::forward<T>(pack)...);
    }
}    // end namespace util
}    // end namespace hpx

#endif    // HPX_UTIL_PACK_TRAVERSAL_ASYNC_HPP
