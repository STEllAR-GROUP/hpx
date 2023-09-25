//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/*
    This test checks that the relocate_at function works correctly
    for non-trivially relocatable types.

    The trivially relocatable optimization can not be implemented yet
    so it is not tested separately.
*/

#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/type_support/construct_at.hpp>

#include <hpx/type_support/is_trivially_relocatable.hpp>
#include <hpx/type_support/relocate_at.hpp>

using hpx::experimental::is_trivially_relocatable_v;
using hpx::experimental::relocate;

struct non_trivially_relocatable_struct
{
    static int count;
    int data;

    explicit non_trivially_relocatable_struct(int data)
      : data(data)
    {
        count++;
    }
    non_trivially_relocatable_struct(non_trivially_relocatable_struct&& other)
      : data(other.data)
    {
        count++;
    }
    ~non_trivially_relocatable_struct()
    {
        count--;
    }

    // making sure the address is never directly accessed
    friend void operator&(non_trivially_relocatable_struct) = delete;
};
int non_trivially_relocatable_struct::count = 0;

static_assert(!is_trivially_relocatable_v<non_trivially_relocatable_struct>);

int hpx_main()
{
    void* mem1 = std::malloc(sizeof(non_trivially_relocatable_struct));

    HPX_TEST(mem1);

    HPX_TEST(non_trivially_relocatable_struct::count == 0);

    non_trivially_relocatable_struct* ptr1 = hpx::construct_at(
        static_cast<non_trivially_relocatable_struct*>(mem1), 1234);

    // a single object was constructed
    HPX_TEST(non_trivially_relocatable_struct::count == 1);

    non_trivially_relocatable_struct obj2 = relocate(ptr1);

    // count = 1 + 1 (from the move construction) - 1 (from the destruction)
    HPX_TEST(non_trivially_relocatable_struct::count == 1);
    HPX_TEST(obj2.data == 1234);

    std::free(mem1);

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    hpx::local::init(hpx_main, argc, argv);
    return hpx::util::report_errors();
}
