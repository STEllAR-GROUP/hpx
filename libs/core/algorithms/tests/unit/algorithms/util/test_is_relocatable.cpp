//  Copyright (c) 2015-2017 Francisco Jose Tapia
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithms/traits/is_relocatable.hpp>
#include <hpx/algorithms/traits/pointer_category.hpp>

#include <cassert>

#include <mutex>

// Integral types are relocatable
static_assert(hpx::relocatable<int>);
static_assert(hpx::relocatable<const int>);
static_assert(hpx::relocatable<int*>);
static_assert(hpx::relocatable<int(*)()>);

// Array types are not move-constructible and thus not relocatable
static_assert(!hpx::relocatable<int[]>);
static_assert(!hpx::relocatable<const int[]>);
static_assert(!hpx::relocatable<int[4]>);
static_assert(!hpx::relocatable<const int[4]>);

// Function types are not move-constructible and thus not relocatable
static_assert(!hpx::relocatable<int()>);

// Void types are not move-constructible and thus not relocatable
static_assert(!hpx::relocatable<void>);
static_assert(!hpx::relocatable<const void>);

// std::mutex is not relocatable
static_assert(!hpx::relocatable<std::mutex>);

struct NotDestructible {
    NotDestructible(const NotDestructible&);
    NotDestructible(NotDestructible&&);
    ~NotDestructible() = delete;
};

static_assert(!hpx::relocatable<NotDestructible>);

struct NotMoveConstructible {
    NotMoveConstructible(const NotMoveConstructible&);
    NotMoveConstructible(NotMoveConstructible&&) = delete;
};

static_assert(!hpx::relocatable<NotMoveConstructible>);

struct NotCopyConstructible {
    NotCopyConstructible(const NotCopyConstructible&) = delete;
    NotCopyConstructible(NotCopyConstructible&&);
};

static_assert(hpx::relocatable<NotCopyConstructible>);

// reference types are relocatable
static_assert(hpx::relocatable<int&>);
static_assert(hpx::relocatable<int&&>);
static_assert(hpx::relocatable<int(&)()>);
static_assert(hpx::relocatable<std::mutex&>);
static_assert(std::relocatable<NotMoveConstructible&>);
static_assert(std::relocatable<NotCopyConstructible&>);
static_assert(std::relocatable<NotDestructible&>);

int main(int, char*[])
{
}
