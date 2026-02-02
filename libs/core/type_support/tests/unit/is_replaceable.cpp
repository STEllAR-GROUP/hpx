//  Copyright (c) 2025 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/type_support/is_replaceable.hpp>

#include <cassert>
#include <memory>
#include <mutex>
#include <type_traits>

using hpx::experimental::is_replaceable_v;

// Integral types are replaceable
static_assert(is_replaceable_v<int>);
// Const types are not assignable
static_assert(!is_replaceable_v<int const>);

// Pointer types are replaceable
static_assert(is_replaceable_v<int*>);
static_assert(is_replaceable_v<int const*>);

// Function pointers are replaceable
static_assert(is_replaceable_v<int (*)()>);

// Arrays are not assignable
static_assert(!is_replaceable_v<int[]>);
static_assert(!is_replaceable_v<int[4]>);

// References are not objects
static_assert(!is_replaceable_v<int&>);
static_assert(!is_replaceable_v<int&&>);

// Void types
static_assert(!is_replaceable_v<void>);

// std::mutex is not move assignable
static_assert(!is_replaceable_v<std::mutex>);

struct not_destructible
{
    not_destructible(not_destructible const&);
    not_destructible(not_destructible&&);
    ~not_destructible() = delete;
};
static_assert(!is_replaceable_v<not_destructible>);

struct not_move_assignable
{
    not_move_assignable& operator=(not_move_assignable&&) = delete;
};
static_assert(!is_replaceable_v<not_move_assignable>);

struct not_move_constructible
{
    not_move_constructible(not_move_constructible&&) = delete;
    not_move_constructible& operator=(not_move_constructible&&);
};
static_assert(!is_replaceable_v<not_move_constructible>);

struct move_assignable
{
    move_assignable(move_assignable&&);
    move_assignable& operator=(move_assignable&&);
    ~move_assignable();
};
static_assert(is_replaceable_v<move_assignable>);

int main() {}
