//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2016 Agustin Berge
//  Copyright (c) 2017 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_DETAIL_RESERVE_HPP
#define HPX_TRAITS_DETAIL_RESERVE_HPP

#include <hpx/config.hpp>
#include <hpx/util/range.hpp>
#include <hpx/traits/is_range.hpp>
#include <hpx/traits/has_member_xxx.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>

namespace hpx { namespace traits { namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    // not every random access sequence is reservable
    // so we need an explicit trait to determine this
    HPX_HAS_MEMBER_XXX_TRAIT_DEF(reserve);

    template <typename Range>
    using is_reservable = std::integral_constant<bool,
          is_range<typename std::decay<Range>::type>::value &&
          has_reserve<typename std::decay<Range>::type>::value>;

    ///////////////////////////////////////////////////////////////////////
    template <typename Container>
    HPX_FORCEINLINE
    typename std::enable_if<!is_reservable<Container>::value>::type
    reserve_if_reservable(Container&, std::size_t) noexcept
    {
    }

    template <typename Container>
    HPX_FORCEINLINE
    typename std::enable_if<is_reservable<Container>::value>::type
    reserve_if_reservable(Container& v, std::size_t n)
    {
        v.reserve(n);
    }

    ///////////////////////////////////////////////////////////////////////
    // Reserve sufficient space in the given vector if the underlying
    // iterator type of the given range allow calculating the size on O(1).
    template <typename Future, typename Range>
    HPX_FORCEINLINE
    void reserve_if_random_access_by_range(std::vector<Future>&,
            Range const&, std::false_type) noexcept
    {
    }

    template <typename Future, typename Range>
    HPX_FORCEINLINE
    void reserve_if_random_access_by_range(std::vector<Future>& v,
            Range const& r, std::true_type)
    {
        v.reserve(util::size(r));
    }

    template <typename Future, typename Range>
    HPX_FORCEINLINE
    void reserve_if_random_access_by_range(std::vector<Future>& v,
            Range const& r)
    {
        typedef typename range_traits<Range>::iterator_category
            iterator_category;

        typedef typename std::is_base_of<
                std::random_access_iterator_tag, iterator_category
            >::type is_random_access;

        reserve_if_random_access_by_range(v, r, is_random_access());
    }

    template <typename Container, typename Range>
    HPX_FORCEINLINE
    void reserve_if_random_access_by_range(Container&, Range const&)
    {
        // do nothing if it's not a vector
    }
}}}

#endif /*HPX_TRAITS_DETAIL_RESERVE_HPP*/
