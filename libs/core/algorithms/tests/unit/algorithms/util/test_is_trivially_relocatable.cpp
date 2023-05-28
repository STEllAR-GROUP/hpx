//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithms/traits/is_trivially_relocatable.hpp>

#include <cassert>

struct TriviallyRelocatable
{
};

HPX_DECLARE_TRIVIALLY_RELOCATABLE(TriviallyRelocatable);

struct NotTriviallyRelocatable
{
};

static_assert(hpx::is_trivially_relocatable_v<TriviallyRelocatable>);
static_assert(!hpx::is_trivially_relocatable_v<NotTriviallyRelocatable>);

// C++ standard library types
static_assert(hpx::is_trivially_relocatable_v<int>);
static_assert(hpx::is_trivially_relocatable_v<double>);
static_assert(hpx::is_trivially_relocatable_v<char>);
static_assert(hpx::is_trivially_relocatable_v<bool>);

// pointers

static_assert(hpx::is_trivially_relocatable_v<void*>);
static_assert(hpx::is_trivially_relocatable_v<int*>);
static_assert(hpx::is_trivially_relocatable_v<double*>);
static_assert(hpx::is_trivially_relocatable_v<NotTriviallyRelocatable*>);

int main(int, char*[]) {}
