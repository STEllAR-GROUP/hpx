//  Copyright (c) 2017 Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_PACK_TRAVERSAL_HPP
#define HPX_UTIL_PACK_TRAVERSAL_HPP

#include <hpx/util/detail/pack_traversal_impl.hpp>

#include <type_traits>
#include <utility>

#if defined(DOXYGEN)
namespace hpx {
namespace util {
    /// Maps the pack with the given mapper.
    ///
    /// This function tries to visit all plain elements which may be wrapped in:
    /// - homogeneous containers (`std::vector`, `std::list`)
    /// - heterogenous containers `(hpx::tuple`, `std::pair`, `std::array`)
    /// and re-assembles the pack with the result of the mapper.
    /// Mapping from one type to a different one is supported.
    ///
    /// Elements that aren't accepted by the mapper are routed through
    /// and preserved through the hierarchy.
    ///
    ///    ```cpp
    ///    // Maps all integers to floats
    ///    map_pack([](int value) {
    ///        return float(value);
    ///    },
    ///    1, hpx::util::make_tuple(2, std::vector<int>{3, 4}), 5);
    ///    ```
    ///
    /// \throws       std::exception like objects which are thrown by an
    ///               invocation to the mapper.
    ///
    /// \param mapper A callable object, which accept an arbitrary type
    ///               and maps it to another type or the same one.
    ///
    /// \param pack   An arbitrary variadic pack which may contain any type.
    ///
    /// \returns      The mapped element or in case the pack contains
    ///               multiple elements, the pack is wrapped into
    ///               a `hpx::tuple`.
    ///
    template <typename Mapper, typename... T>
    <unspecified> map_pack(Mapper&& mapper, T&&... pack);
}
}
#else // DOXYGEN

namespace hpx {
namespace util {
    template <typename Mapper, typename... T>
    auto map_pack(Mapper&& mapper, T&&... pack)
        -> decltype(detail::apply_pack_transform(detail::strategy_remap_tag{},
            std::forward<Mapper>(mapper),
            std::forward<T>(pack)...))
    {
        return detail::apply_pack_transform(detail::strategy_remap_tag{},
            std::forward<Mapper>(mapper),
            std::forward<T>(pack)...);
    }

    /// Indicate that the result shall be spread across the parent container
    /// if possible. This can be used to create a mapper function used
    /// in map_pack that maps one element to an arbitrary count (1:n).
    template <typename... T>
    detail::spreading::spread_box<typename std::decay<T>::type...> spread_this(
        T&&... args)
    {
        return detail::spreading::spread_box<typename std::decay<T>::type...>(
            make_tuple(std::forward<T>(args)...));
    }

    /// Traverses the pack with the given visitor.
    ///
    /// This function works in the same way as `map_pack`,
    /// however, the result of the mapper isn't preserved.
    ///
    /// See `map_pack` for a detailed description.
    template <typename Mapper, typename... T>
    void traverse_pack(Mapper&& mapper, T&&... pack)
    {
        detail::apply_pack_transform(detail::strategy_traverse_tag{},
            std::forward<Mapper>(mapper),
            std::forward<T>(pack)...);
    }
}    // end namespace util
}    // end namespace hpx

#endif  // DOXYGEN
#endif  // HPX_UTIL_PACK_TRAVERSAL_HPP
