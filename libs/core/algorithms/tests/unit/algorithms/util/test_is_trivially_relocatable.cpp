//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithms/traits/is_trivially_relocatable.hpp>

#include <cassert>

// Trivially Copyable types are trivially relocatable
struct trivially_copyable
{
    trivially_copyable(trivially_copyable const&) = default;
};
static_assert(hpx::is_trivially_relocatable_v<trivially_copyable>,
    "Trivially Copyable type should be Trivially Relocatable");

// Non trivially copyable types are not trivially relocatable
// Unless they are explicitly declared as such
struct not_trivially_copyable_1
{
    not_trivially_copyable_1(not_trivially_copyable_1 const&){};
};
static_assert(!hpx::is_trivially_relocatable_v<not_trivially_copyable_1>,
    "Not Trivially Copyable and Not declared Trivially Relocatable type should "
    "not be Trivially Relocatable");

struct not_trivially_copyable_2
{
    not_trivially_copyable_2(not_trivially_copyable_2 const&){};
};
HPX_DECLARE_TRIVIALLY_RELOCATABLE(not_trivially_copyable_2)

static_assert(hpx::is_trivially_relocatable_v<not_trivially_copyable_2>,
    "Not Trivially Copyable but declared Trivially Relocatable type should "
    "be Trivially Relocatable");

// Primive data types are trivially relocatable
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
