//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithms/traits/is_trivially_relocatable.hpp>

#include <cassert>

// Trivially Copyable types are trivially relocatable
struct TriviallyCopyable
{
    TriviallyCopyable(TriviallyCopyable const&) = default;
};
static_assert(hpx::is_trivially_relocatable_v<TriviallyCopyable>,
    "Trivially Copyable type should be Trivially Relocatable");

// Non trivially copyable types are not trivially relocatable
// Unless they are explicitly declared as such
struct NotTriviallyCopyable_1
{
    NotTriviallyCopyable_1(NotTriviallyCopyable_1 const&){};
};
static_assert(!hpx::is_trivially_relocatable_v<NotTriviallyCopyable_1>,
    "Not Trivially Copyable and Not declared Trivially Relocatable type should "
    "not be Trivially Relocatable");

struct NotTriviallyCopyable_2
{
    NotTriviallyCopyable_2(NotTriviallyCopyable_2 const&){};
};
HPX_DECLARE_TRIVIALLY_RELOCATABLE(NotTriviallyCopyable_2)

static_assert(hpx::is_trivially_relocatable_v<NotTriviallyCopyable_2>,
    "Not Trivially Copyable but declared Trivially Relocatable type should "
    "be Trivially Relocatable");

// Standard library types are trivially relocatable
static_assert(hpx::is_trivially_relocatable_v<int>,
    "int should be Trivially Relocatable");
static_assert(hpx::is_trivially_relocatable_v<double>,
    "double should be Trivially Relocatable");
static_assert(hpx::is_trivially_relocatable_v<char>,
    "char should be Trivially Relocatable");
static_assert(hpx::is_trivially_relocatable_v<void*>,
    "void* should be Trivially Relocatable");
static_assert(hpx::is_trivially_relocatable_v<int*>,
    "int* should be Trivially Relocatable");
static_assert(hpx::is_trivially_relocatable_v<double*>,
    "double* should be Trivially Relocatable");
static_assert(hpx::is_trivially_relocatable_v<char*>,
    "char* should be Trivially Relocatable");

int main(int, char*[]) {}
