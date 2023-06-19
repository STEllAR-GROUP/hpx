//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/algorithms/traits/is_trivially_relocatable.hpp>

#include <cassert>

// Trivially copyable types are trivially relocatable
struct empty
{
};

struct non_empty_but_trivial
{
    int i;
};

struct non_trivial_but_trivially_copyable
{
    explicit non_trivial_but_trivially_copyable(int);
    int i;
};

struct move_only_but_trivially_copyable
{
    int i;
    explicit move_only_but_trivially_copyable(int);
    move_only_but_trivially_copyable(
        move_only_but_trivially_copyable&&) noexcept = default;
    move_only_but_trivially_copyable& operator=(
        move_only_but_trivially_copyable&&) noexcept = default;
    ~move_only_but_trivially_copyable() = default;
};

struct non_assignable_but_trivially_copyable
{
    int i;
    explicit non_assignable_but_trivially_copyable(int);
    non_assignable_but_trivially_copyable(
        non_assignable_but_trivially_copyable&&) = default;
    non_assignable_but_trivially_copyable operator=(
        non_assignable_but_trivially_copyable&&) = delete;
    ~non_assignable_but_trivially_copyable() = default;
};

static_assert(hpx::is_trivially_relocatable_v<empty>);
static_assert(hpx::is_trivially_relocatable_v<non_empty_but_trivial>);
static_assert(
    hpx::is_trivially_relocatable_v<non_trivial_but_trivially_copyable>);
static_assert(
    hpx::is_trivially_relocatable_v<move_only_but_trivially_copyable>);
static_assert(
    hpx::is_trivially_relocatable_v<non_assignable_but_trivially_copyable>);

// Has non trivial copy constructor
struct not_trivially_copyable_1
{
    not_trivially_copyable_1();
    not_trivially_copyable_1(not_trivially_copyable_1 const&);
    not_trivially_copyable_1& operator=(
        not_trivially_copyable_1 const&) = default;
    ~not_trivially_copyable_1() = default;
};

// Has non trivial copy assignment
struct not_trivially_copyable_2
{
    not_trivially_copyable_2();
    not_trivially_copyable_2(not_trivially_copyable_2 const&) = default;
    not_trivially_copyable_2& operator=(not_trivially_copyable_2 const&);
    ~not_trivially_copyable_2() = default;
};

// Has non trivial destructor
struct not_trivially_copyable_3
{
    not_trivially_copyable_3();
    not_trivially_copyable_3(not_trivially_copyable_3 const&) = default;
    not_trivially_copyable_3& operator=(
        not_trivially_copyable_3 const&) = default;
    ~not_trivially_copyable_3();
};

// Non trivially copyable types are not trivially relocatable
// Unless they are explicitly declared as such
static_assert(!hpx::is_trivially_relocatable_v<not_trivially_copyable_1>);
static_assert(!hpx::is_trivially_relocatable_v<not_trivially_copyable_2>);
static_assert(!hpx::is_trivially_relocatable_v<not_trivially_copyable_2>);

// Not trivially copyable
struct explicitly_trivially_relocatable_1
{
    explicitly_trivially_relocatable_1() = default;
    explicitly_trivially_relocatable_1(
        explicitly_trivially_relocatable_1 const&);
    explicitly_trivially_relocatable_1& operator=(
        explicitly_trivially_relocatable_1 const&);
    ~explicitly_trivially_relocatable_1() = default;
};

// Has non trivial copy assignment
struct explicitly_trivially_relocatable_2
{
    explicitly_trivially_relocatable_2() = default;
    explicitly_trivially_relocatable_2(explicitly_trivially_relocatable_2&&);
    explicitly_trivially_relocatable_2& operator=(
        explicitly_trivially_relocatable_2&&);
    ~explicitly_trivially_relocatable_2() = default;
};

HPX_DECLARE_TRIVIALLY_RELOCATABLE(explicitly_trivially_relocatable_1);
HPX_DECLARE_TRIVIALLY_RELOCATABLE(explicitly_trivially_relocatable_2);

// Explicitly declared trivially relocatable types are trivially relocatable
static_assert(
    hpx::is_trivially_relocatable_v<explicitly_trivially_relocatable_1>);
static_assert(
    hpx::is_trivially_relocatable_v<explicitly_trivially_relocatable_2>);

// Trivial relocatability is not inherited
struct derived_from_explicitly_trivially_relocatable
  : explicitly_trivially_relocatable_1
{
};

static_assert(!hpx::is_trivially_relocatable_v<
              derived_from_explicitly_trivially_relocatable>);

// Polymorphic types are not trivially relocatable
struct polymorphic
{
    virtual int f();
};
static_assert(!hpx::is_trivially_relocatable_v<polymorphic>);

// Primitive data types are trivially relocatable
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
