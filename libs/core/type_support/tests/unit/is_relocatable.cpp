//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/type_support/is_relocatable.hpp>

#include <cassert>

#include <mutex>

// Integral types are relocatable
static_assert(hpx::is_relocatable_v<int>);
static_assert(hpx::is_relocatable_v<int const>);
static_assert(hpx::is_relocatable_v<int*>);
static_assert(hpx::is_relocatable_v<int (*)()>);

// Array types are not move-constructible and thus not relocatable
static_assert(!hpx::is_relocatable_v<int[]>);
static_assert(!hpx::is_relocatable_v<int const[]>);
static_assert(!hpx::is_relocatable_v<int[4]>);
static_assert(!hpx::is_relocatable_v<int const[4]>);

// Function types are not move-constructible and thus not relocatable
static_assert(!hpx::is_relocatable_v<int()>);

// Void types are not move-constructible and thus not relocatable
static_assert(!hpx::is_relocatable_v<void>);
static_assert(!hpx::is_relocatable_v<void const>);

// std::mutex is not relocatable
static_assert(!hpx::is_relocatable_v<std::mutex>);

struct not_destructible
{
    not_destructible(not_destructible const&);
    not_destructible(not_destructible&&);
    ~not_destructible() = delete;
};

static_assert(!hpx::is_relocatable_v<not_destructible>);
struct not_move_constructible
{
    not_move_constructible(not_move_constructible const&);
    not_move_constructible(not_move_constructible&&) = delete;
};

static_assert(!hpx::is_relocatable_v<not_move_constructible>);

struct not_copy_constructible
{
    not_copy_constructible(not_copy_constructible const&) = delete;
    not_copy_constructible(not_copy_constructible&&);
};

static_assert(hpx::is_relocatable_v<not_copy_constructible>);

// reference types are relocatable
static_assert(hpx::is_relocatable_v<int&>);
static_assert(hpx::is_relocatable_v<int&&>);
static_assert(hpx::is_relocatable_v<int (&)()>);
static_assert(hpx::is_relocatable_v<std::mutex&>);
static_assert(hpx::is_relocatable_v<not_move_constructible&>);
static_assert(hpx::is_relocatable_v<not_copy_constructible&>);
static_assert(hpx::is_relocatable_v<not_destructible&>);

int main(int, char*[]) {}
