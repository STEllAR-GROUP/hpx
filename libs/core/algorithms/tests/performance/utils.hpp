///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hpx/modules/compute.hpp>

#include <cstddef>
#include <forward_list>
#include <iterator>
#include <list>
#include <type_traits>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
template <typename IterTag, typename DataType = int,
    typename Alloc = std::allocator<DataType>, typename Enable = void>
struct test_container;

template <typename RandIterTag, typename DataType>
struct test_container<RandIterTag, DataType, std::allocator<DataType>,
    std::enable_if_t<
        std::is_same_v<RandIterTag, std::random_access_iterator_tag>>>
{
    using type = std::vector<DataType, std::allocator<DataType>>;

    static type get_container(std::size_t size,
        std::allocator<DataType> const& alloc = std::allocator<DataType>{})
    {
        return type(size, alloc);
    }
};

template <typename RandIterTag, typename DataType, typename Alloc>
struct test_container<RandIterTag, DataType, Alloc,
    std::enable_if_t<
        std::is_same_v<RandIterTag, std::random_access_iterator_tag>>>
{
    using type = hpx::compute::vector<DataType, Alloc>;

    static type get_container(std::size_t size, Alloc const& alloc = Alloc{})
    {
        return type(size, alloc);
    }
};

template <typename BidirIterTag, typename DataType, typename Alloc>
struct test_container<BidirIterTag, DataType, Alloc,
    std::enable_if_t<
        std::is_same_v<BidirIterTag, std::bidirectional_iterator_tag>>>
{
    using type = std::list<DataType, Alloc>;

    static type get_container(std::size_t size, Alloc const& alloc = Alloc{})
    {
        return type(size, alloc);
    }
};

template <typename FwdIterTag, typename DataType, typename Alloc>
struct test_container<FwdIterTag, DataType, Alloc,
    std::enable_if_t<std::is_same_v<FwdIterTag, std::forward_iterator_tag>>>
{
    using type = std::forward_list<DataType, Alloc>;

    static type get_container(std::size_t size, Alloc const& alloc = Alloc{})
    {
        return type(size, alloc);
    }
};
