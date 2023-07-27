//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/type_support/construct_at.hpp>
#include <hpx/type_support/relocate_at.hpp>

struct non_trivially_relocatable_struct
{
    static int count;
    int data;

    non_trivially_relocatable_struct(int data)
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

static_assert(
    !hpx::is_trivially_relocatable_v<non_trivially_relocatable_struct>);


struct trivially_relocatable_struct
{
    static int count;
    int data;

    trivially_relocatable_struct(int data)
      : data(data)
    {
        count++;
    }
    trivially_relocatable_struct(trivially_relocatable_struct&& other)
      : data(other.data)
    {
        count++;
    }
    ~trivially_relocatable_struct()
    {
        count--;
    }

    friend void operator&(trivially_relocatable_struct) = delete;
};
int trivially_relocatable_struct::count = 0;

HPX_DECLARE_TRIVIALLY_RELOCATABLE(trivially_relocatable_struct);
static_assert(hpx::is_trivially_relocatable_v<trivially_relocatable_struct>);

int hpx_main()
{
    {
        void* mem1 = std::malloc(sizeof(non_trivially_relocatable_struct));
        void* mem2 = std::malloc(sizeof(non_trivially_relocatable_struct));

        HPX_TEST(mem1 && mem2);

        HPX_TEST(non_trivially_relocatable_struct::count == 0);

        non_trivially_relocatable_struct* ptr1 = hpx::construct_at(
            static_cast<non_trivially_relocatable_struct*>(mem1), 1234);

        non_trivially_relocatable_struct* ptr2 =
            static_cast<non_trivially_relocatable_struct*>(mem2);

        // a single object was constructed
        HPX_TEST(non_trivially_relocatable_struct::count == 1);

        hpx::relocate_at(ptr1, ptr2);

        // count = 1 + 1 (from the move construction) - 1 (from the destruction)
        HPX_TEST(non_trivially_relocatable_struct::count == 1);
        HPX_TEST(ptr2->data == 1234);

        std::destroy_at(ptr2);

        std::free(mem1);
        std::free(mem2);
    }
    {
        void* mem1 = std::malloc(sizeof(trivially_relocatable_struct));
        void* mem2 = std::malloc(sizeof(trivially_relocatable_struct));

        HPX_TEST(mem1 && mem2);

        HPX_TEST(trivially_relocatable_struct::count == 0);

        trivially_relocatable_struct* ptr1 = hpx::construct_at(
            static_cast<trivially_relocatable_struct*>(mem1), 1234);

        trivially_relocatable_struct* ptr2 =
            static_cast<trivially_relocatable_struct*>(mem2);

        // a single object was constructed
        HPX_TEST(trivially_relocatable_struct::count == 1);

        hpx::relocate_at(ptr1, ptr2);

        // count = 1 + 0 (relocation on trivially relocatable objects does not trigger move constructors
        // or destructors); no object is destroyed or created
        HPX_TEST(trivially_relocatable_struct::count == 1);
        HPX_TEST(ptr2->data == 1234);

        std::destroy_at(ptr2);

        std::free(mem1);
        std::free(mem2);
    }
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    hpx::local::init(hpx_main, argc, argv);
    return hpx::util::report_errors();
}