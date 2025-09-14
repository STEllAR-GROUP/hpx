//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/type_support.hpp>

#include <cassert>

#include <memory>    // for std::shared_ptr, std::unique_ptr
#include <mutex>

using hpx::experimental::is_relocatable_from_v;
using hpx::experimental::is_relocatable_v;

// Integral types are relocatable
static_assert(is_relocatable_v<int>);
static_assert(is_relocatable_v<int const>);

// Pointer types are relocatable
static_assert(is_relocatable_v<int*>);
static_assert(is_relocatable_v<int (*)()>);
static_assert(is_relocatable_v<int (*)[]>);
static_assert(is_relocatable_v<int (*)[4]>);

// Array types are not move-constructible and thus not relocatable
static_assert(!is_relocatable_v<int[]>);
static_assert(!is_relocatable_v<int[4]>);
static_assert(!is_relocatable_v<int const[]>);
static_assert(!is_relocatable_v<int const[4]>);

// Function types are not move-constructible and thus not relocatable
static_assert(!is_relocatable_v<int()>);

// Void types are not move-constructible and thus not relocatable
static_assert(!is_relocatable_v<void>);
static_assert(!is_relocatable_v<void const>);

// std::mutex is not relocatable
static_assert(!is_relocatable_v<std::mutex>);

struct not_destructible
{
    not_destructible(not_destructible const&);
    not_destructible(not_destructible&&);
    ~not_destructible() = delete;
};

static_assert(!is_relocatable_v<not_destructible>);
struct not_move_constructible
{
    not_move_constructible(not_move_constructible const&);
    not_move_constructible(not_move_constructible&&) = delete;
};

static_assert(!is_relocatable_v<not_move_constructible>);

struct not_copy_constructible
{
    not_copy_constructible(not_copy_constructible const&) = delete;
    not_copy_constructible(not_copy_constructible&&);
};

static_assert(is_relocatable_v<not_copy_constructible>);

// reference types are not relocatable
static_assert(!is_relocatable_v<int&>);
static_assert(!is_relocatable_v<int&&>);
static_assert(!is_relocatable_v<int (&)()>);
static_assert(!is_relocatable_v<std::mutex&>);
static_assert(!is_relocatable_v<not_move_constructible&>);
static_assert(!is_relocatable_v<not_copy_constructible&>);
static_assert(!is_relocatable_v<not_destructible&>);

/*
    Tests for is_relocatable_from
*/

// clang-format off

// Reference types are not relocatable
static_assert(!is_relocatable_from_v<
    int (&)[], int (&)[4]>);

// Array types are not move constructible
static_assert(!is_relocatable_from_v<
    int[4], int[4]>);

// This is a simple pointer
static_assert(is_relocatable_from_v<
    int (*)[4], int (*)[4]>);

// Can move from const shared_ptr
static_assert(is_relocatable_from_v<
    std::shared_ptr<int>, const std::shared_ptr<int>>);

// Can't move away from a const unique_ptr
static_assert(!is_relocatable_from_v<
    std::unique_ptr<int>, const std::unique_ptr<int>>);

// Can move away from a non-const unique_ptr
static_assert(is_relocatable_from_v<
    std::unique_ptr<int>, std::unique_ptr<int>>);

// Can move away from a non-const unique_ptr, the dest's constness does not matter
static_assert(is_relocatable_from_v<
    const std::unique_ptr<int>, std::unique_ptr<int>>);

// Can't move away from a const unique_ptr, the dest's constness does not matter
static_assert(!is_relocatable_from_v<
    const std::unique_ptr<int>, const std::unique_ptr<int>>);
// clang-format on

int main(int, char*[]) {}
