//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/type_support/is_contiguous_iterator.hpp>

#include <array>
#include <cassert>
#include <iterator>
#include <list>
#include <set>
#include <string>
#include <vector>

using namespace hpx::traits::detail;
using namespace hpx::traits;

// std::vector<int>::iterator is a contiguous iterator
static_assert(has_valid_vector_v<int>);

static_assert(is_contiguous_iterator_v<std::vector<int>::iterator>);
static_assert(is_std_vector_iterator<std::vector<int>::iterator>::value);

static_assert(is_contiguous_iterator_v<std::vector<int>::const_iterator>);
static_assert(is_std_vector_iterator<std::vector<int>::const_iterator>::value);

// reverse_iterator is not a contiguous iterator
static_assert(!is_contiguous_iterator_v<std::vector<int>::reverse_iterator>);
static_assert(
    !is_contiguous_iterator_v<std::vector<int>::const_reverse_iterator>);

// std::array<int, 4>::iterator is a contiguous iterator
static_assert(is_contiguous_iterator_v<std::array<int, 1>::iterator>);
static_assert(is_contiguous_iterator_v<std::array<int, 1>::const_iterator>);
// reverse_iterator is not a contiguous iterator
static_assert(!is_contiguous_iterator_v<std::array<int, 1>::reverse_iterator>);
static_assert(
    !is_contiguous_iterator_v<std::array<int, 1>::const_reverse_iterator>);

// pointers are contiguous iterators
static_assert(is_contiguous_iterator_v<int*>);
static_assert(is_contiguous_iterator_v<int const*>);

// std::string::iterator is a contiguous iterator
static_assert(is_contiguous_iterator_v<std::string::iterator>);
static_assert(is_contiguous_iterator_v<std::string::const_iterator>);

// Make sure #6754 was properly fixed
static_assert(!is_contiguous_iterator_v<std::list<char>::iterator>);
static_assert(!is_contiguous_iterator_v<std::list<char>::const_iterator>);
static_assert(!is_contiguous_iterator_v<std::set<char>::iterator>);
static_assert(!is_contiguous_iterator_v<std::set<char>::const_iterator>);

// Pointers to arrays are still pointers
static_assert(is_contiguous_iterator_v<int (*)[]>);
static_assert(is_contiguous_iterator_v<int const (*)[]>);
static_assert(is_contiguous_iterator_v<int (*)[4]>);
static_assert(is_contiguous_iterator_v<int const (*)[4]>);

// Pointers to functions are still pointers
static_assert(is_contiguous_iterator_v<int (*)()>);
static_assert(is_contiguous_iterator_v<int const (*)()>);

// c-style arrays are not contiguous iterators
static_assert(!is_contiguous_iterator_v<int[]>);
static_assert(!is_contiguous_iterator_v<int[4]>);
static_assert(!is_contiguous_iterator_v<int const[]>);
static_assert(!is_contiguous_iterator_v<int const[4]>);

int main(int, char*[]) {}
