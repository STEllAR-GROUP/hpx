//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <array>
#include <cassert>
#include <vector>

#include <hpx/type_support/is_contiguous_iterator.hpp>

using hpx::traits::is_contiguous_iterator_v;

// std::vector<int>::iterator is a contiguous iterator
static_assert(is_contiguous_iterator_v<std::vector<int>::iterator>);
static_assert(is_contiguous_iterator_v<std::vector<int>::const_iterator>);
// reverse_iterator is not a contiguous iterator
static_assert(!is_contiguous_iterator_v<std::vector<int>::reverse_iterator>);
static_assert(
    !is_contiguous_iterator_v<std::vector<int>::const_reverse_iterator>);

// std::array<int, 4>::iterator is a contiguous iterator
static_assert(is_contiguous_iterator_v<std::array<int, 4>::iterator>);
static_assert(is_contiguous_iterator_v<std::array<int, 4>::const_iterator>);
// reverse_iterator is not a contiguous iterator
static_assert(!is_contiguous_iterator_v<std::array<int, 4>::reverse_iterator>);
static_assert(
    !is_contiguous_iterator_v<std::array<int, 4>::const_reverse_iterator>);

// pointers are contiguous iterators
static_assert(is_contiguous_iterator_v<int*>);
static_assert(is_contiguous_iterator_v<int const*>);
static_assert(is_contiguous_iterator_v<int (*)[]>);
static_assert(is_contiguous_iterator_v<int const (*)[]>);
static_assert(is_contiguous_iterator_v<int (*)[4]>);
static_assert(is_contiguous_iterator_v<int const (*)[4]>);

// arrays are not contiguous iterators
static_assert(!is_contiguous_iterator_v<int[]>);
static_assert(!is_contiguous_iterator_v<int[4]>);
static_assert(!is_contiguous_iterator_v<int const[]>);
static_assert(!is_contiguous_iterator_v<int const[4]>);

// std::string::iterator is a contiguous iterator
static_assert(is_contiguous_iterator_v<std::string::iterator>);
static_assert(is_contiguous_iterator_v<std::string::const_iterator>);

int main(int, char*[]) {}
