//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/type_support/is_trivially_relocatable.hpp>
#include <type_traits>

#include <cassert>

using hpx::experimental::is_trivially_relocatable_v;

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

static_assert(is_trivially_relocatable_v<empty>);
static_assert(is_trivially_relocatable_v<non_empty_but_trivial>);
static_assert(is_trivially_relocatable_v<non_trivial_but_trivially_copyable>);
static_assert(is_trivially_relocatable_v<move_only_but_trivially_copyable>);
static_assert(
    is_trivially_relocatable_v<non_assignable_but_trivially_copyable>);

// Non trivially copyable types should not be considered trivially
// relocatable by default

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

static_assert(!is_trivially_relocatable_v<not_trivially_copyable_1>);
static_assert(!is_trivially_relocatable_v<not_trivially_copyable_2>);
static_assert(!is_trivially_relocatable_v<not_trivially_copyable_2>);

// The HPX_DECLARE_TRIVIALLY_RELOCATABLE macro declares types as trivially
// relocatable

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
static_assert(is_trivially_relocatable_v<explicitly_trivially_relocatable_1>);
static_assert(is_trivially_relocatable_v<explicitly_trivially_relocatable_2>);

// c-v-ref-array qualified versions of explicitly declared trivially relocatable
// types are trivially relocatable

static_assert(
    is_trivially_relocatable_v<explicitly_trivially_relocatable_1 const>);
static_assert(
    is_trivially_relocatable_v<explicitly_trivially_relocatable_1 volatile>);
static_assert(is_trivially_relocatable_v<
    explicitly_trivially_relocatable_1 const volatile>);
static_assert(is_trivially_relocatable_v<explicitly_trivially_relocatable_1[]>);
static_assert(
    is_trivially_relocatable_v<explicitly_trivially_relocatable_1[10]>);

// Chain of c-v-array qualifiers are supported
static_assert(
    is_trivially_relocatable_v<explicitly_trivially_relocatable_1[10][10]>);
static_assert(
    is_trivially_relocatable_v<explicitly_trivially_relocatable_1 const[10]>);
static_assert(is_trivially_relocatable_v<
    explicitly_trivially_relocatable_1 volatile[10]>);
static_assert(is_trivially_relocatable_v<
    explicitly_trivially_relocatable_1 const volatile[10]>);

// References and temporaries are not trivially relocatable
// clang-format off
static_assert(!is_trivially_relocatable_v<
    explicitly_trivially_relocatable_1&>);
static_assert(!is_trivially_relocatable_v<
    explicitly_trivially_relocatable_1&&>);
static_assert(!is_trivially_relocatable_v<
    explicitly_trivially_relocatable_1 (&)[10]>);
static_assert(!is_trivially_relocatable_v<
    explicitly_trivially_relocatable_1 (&&)[10]>);
static_assert(!is_trivially_relocatable_v<
              explicitly_trivially_relocatable_1 const volatile&>);
// clang-format on

// c-v-ref-array qualified versions of explicitly declared trivially relocatable
// types are trivially relocatable

// Trivial relocatability is not inherited
struct derived_from_explicitly_trivially_relocatable
  : explicitly_trivially_relocatable_1
{
};

static_assert(
    !is_trivially_relocatable_v<derived_from_explicitly_trivially_relocatable>);

// Polymorphic types are not trivially relocatable
struct polymorphic
{
    virtual int f();
};
static_assert(!is_trivially_relocatable_v<polymorphic>);

// Test that it is not breaking to declare an already known
// trivially copyable type to be trivially relocatable
struct trivially_copyable_explicitly_trivially_relocatable
{
    int i;
    trivially_copyable_explicitly_trivially_relocatable(
        trivially_copyable_explicitly_trivially_relocatable const&) = default;
    trivially_copyable_explicitly_trivially_relocatable& operator=(
        trivially_copyable_explicitly_trivially_relocatable const&) = default;
    ~trivially_copyable_explicitly_trivially_relocatable() = default;
};
HPX_DECLARE_TRIVIALLY_RELOCATABLE(
    trivially_copyable_explicitly_trivially_relocatable);

static_assert(std::is_trivially_copyable_v<
    trivially_copyable_explicitly_trivially_relocatable>);
static_assert(is_trivially_relocatable_v<
    trivially_copyable_explicitly_trivially_relocatable>);

// Testing the HPX_DECLARE_TRIVIALLY_RELOCATABLE_TEMPLATE macro
template <typename T, typename K>
struct non_trivially_copyable
{
    non_trivially_copyable(non_trivially_copyable const&);
    non_trivially_copyable operator=(non_trivially_copyable const&);
    ~non_trivially_copyable();
};

static_assert(!std::is_trivially_copyable_v<non_trivially_copyable<int, int>>);

HPX_DECLARE_TRIVIALLY_RELOCATABLE_TEMPLATE(non_trivially_copyable);
static_assert(is_trivially_relocatable_v<non_trivially_copyable<int, int>>);

// Testing the HPX_DECLARE_TRIVIALLY_RELOCATABLE_TEMPLATE_IF macro
struct trivially_relocatable_struct
{
};
static_assert(is_trivially_relocatable_v<trivially_relocatable_struct>);

struct non_trivially_relocatable_struct
{
    non_trivially_relocatable_struct(non_trivially_relocatable_struct const&);
};
static_assert(!is_trivially_relocatable_v<non_trivially_relocatable_struct>);

template <typename T, typename K>
struct non_trivially_copyable_container
{
    T t;
    K k;

    non_trivially_copyable_container();
    non_trivially_copyable_container(non_trivially_copyable_container const&);
    non_trivially_copyable_container& operator=(
        non_trivially_copyable_container const&);
    ~non_trivially_copyable_container();
};
// If T is trivially relocatable then non_trivially_copyable_container is
// trivially relocatable too.

template <typename T, typename K>
struct my_metafunction
  : std::bool_constant<is_trivially_relocatable_v<T> &&
        is_trivially_relocatable_v<K>>
{
};

HPX_DECLARE_TRIVIALLY_RELOCATABLE_TEMPLATE_IF(
    non_trivially_copyable_container, my_metafunction)

static_assert(is_trivially_relocatable_v<non_trivially_copyable_container<
        trivially_relocatable_struct, trivially_relocatable_struct>>);
static_assert(!is_trivially_relocatable_v<non_trivially_copyable_container<
        trivially_relocatable_struct, non_trivially_relocatable_struct>>);
static_assert(!is_trivially_relocatable_v<non_trivially_copyable_container<
        non_trivially_relocatable_struct, trivially_relocatable_struct>>);
static_assert(!is_trivially_relocatable_v<non_trivially_copyable_container<
        non_trivially_relocatable_struct, non_trivially_relocatable_struct>>);

// Primitive data types are trivially relocatable
static_assert(
    is_trivially_relocatable_v<int>, "int should be Trivially Relocatable");
static_assert(is_trivially_relocatable_v<double>,
    "double should be Trivially Relocatable");
static_assert(
    is_trivially_relocatable_v<char>, "char should be Trivially Relocatable");
static_assert(
    is_trivially_relocatable_v<void*>, "void* should be Trivially Relocatable");
static_assert(
    is_trivially_relocatable_v<int*>, "int* should be Trivially Relocatable");
static_assert(is_trivially_relocatable_v<double*>,
    "double* should be Trivially Relocatable");
static_assert(
    is_trivially_relocatable_v<char*>, "char* should be Trivially Relocatable");

// Void and function types are not trivially relocatable
static_assert(!is_trivially_relocatable_v<void>);
static_assert(!is_trivially_relocatable_v<int()>);
static_assert(!is_trivially_relocatable_v<int (&)()>);

int main(int, char*[]) {}
