//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithms/traits/is_relocatable.hpp>
#include <hpx/algorithms/traits/pointer_category.hpp>

#include <cassert>

#include <mutex>

// Integral types are relocatable
static_assert(hpx::is_relocatable_v<int>);
static_assert(hpx::is_relocatable_v<const int>);
static_assert(hpx::is_relocatable_v<int*>);
static_assert(hpx::is_relocatable_v<int (*)()>);

// Array types are not move-constructible and thus not relocatable
static_assert(!hpx::is_relocatable_v<int[]>);
static_assert(!hpx::is_relocatable_v<const int[]>);
static_assert(!hpx::is_relocatable_v<int[4]>);
static_assert(!hpx::is_relocatable_v<const int[4]>);

// Function types are not move-constructible and thus not relocatable
static_assert(!hpx::is_relocatable_v<int()>);

// Void types are not move-constructible and thus not relocatable
static_assert(!hpx::is_relocatable_v<void>);
static_assert(!hpx::is_relocatable_v<const void>);

// std::mutex is not relocatable
static_assert(!hpx::is_relocatable_v<std::mutex>);

struct NotDestructible
{
    NotDestructible(const NotDestructible&);
    NotDestructible(NotDestructible&&);
    ~NotDestructible() = delete;
};

static_assert(!hpx::is_relocatable_v<NotDestructible>);

struct NotMoveConstructible
{
    NotMoveConstructible(const NotMoveConstructible&);
    NotMoveConstructible(NotMoveConstructible&&) = delete;
};

static_assert(!hpx::is_relocatable_v<NotMoveConstructible>);

struct NotCopyConstructible
{
    NotCopyConstructible(const NotCopyConstructible&) = delete;
    NotCopyConstructible(NotCopyConstructible&&);
};

static_assert(hpx::is_relocatable_v<NotCopyConstructible>);

// reference types are relocatable
static_assert(hpx::is_relocatable_v<int&>);
static_assert(hpx::is_relocatable_v<int&&>);
static_assert(hpx::is_relocatable_v<int (&)()>);
static_assert(hpx::is_relocatable_v<std::mutex&>);
static_assert(hpx::is_relocatable_v<NotMoveConstructible&>);
static_assert(hpx::is_relocatable_v<NotCopyConstructible&>);
static_assert(hpx::is_relocatable_v<NotDestructible&>);

int main(int, char*[]) {}
