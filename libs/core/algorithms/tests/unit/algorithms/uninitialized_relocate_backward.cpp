//  Copyright (c) 2023 Isidoros Tsaousis-Seiras
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/algorithms/uninitialized_relocate.hpp>

#define N 50
#define M 20
#define K 10

using hpx::experimental::is_trivially_relocatable_v;
using hpx::experimental::uninitialized_relocate_backward;

struct trivially_relocatable_struct
{
    static int count;
    static int move_count;
    static int dtor_count;
    int data;

    explicit trivially_relocatable_struct(int data)
      : data(data)
    {
        count++;
    }
    trivially_relocatable_struct(trivially_relocatable_struct&& other)
      : data(other.data)
    {
        move_count++;
        count++;
    }
    ~trivially_relocatable_struct()
    {
        dtor_count++;
        count--;
    }

    // making sure the address is never directly accessed
    friend void operator&(trivially_relocatable_struct) = delete;
};
int trivially_relocatable_struct::count = 0;
int trivially_relocatable_struct::move_count = 0;
int trivially_relocatable_struct::dtor_count = 0;

HPX_DECLARE_TRIVIALLY_RELOCATABLE(trivially_relocatable_struct);
static_assert(is_trivially_relocatable_v<trivially_relocatable_struct>);

struct non_trivially_relocatable_struct
{
    static int count;
    static int move_count;
    static int dtor_count;
    int data;

    explicit non_trivially_relocatable_struct(int data)
      : data(data)
    {
        count++;
    }
    // mark as noexcept to enter simpler relocation path
    non_trivially_relocatable_struct(
        non_trivially_relocatable_struct&& other) noexcept
      : data(other.data)
    {
        move_count++;
        count++;
    }
    ~non_trivially_relocatable_struct()
    {
        dtor_count++;
        count--;
    }

    // making sure the address is never directly accessed
    friend void operator&(non_trivially_relocatable_struct) = delete;
};
int non_trivially_relocatable_struct::count = 0;
int non_trivially_relocatable_struct::move_count = 0;
int non_trivially_relocatable_struct::dtor_count = 0;

static_assert(!is_trivially_relocatable_v<non_trivially_relocatable_struct>);

struct non_trivially_relocatable_struct_throwing
{
    static int count;
    static int move_count;
    static int dtor_count;

    int data;

    explicit non_trivially_relocatable_struct_throwing(int data)
      : data(data)
    {
        count++;
    }
    // do not mark as noexcept to enter try-catch relocation path
    non_trivially_relocatable_struct_throwing(
        non_trivially_relocatable_struct_throwing&& other)
      : data(other.data)
    {
        if (move_count == K)
        {
            throw 42;
        }
        move_count++;

        count++;
    }
    ~non_trivially_relocatable_struct_throwing()
    {
        dtor_count++;
        count--;
    }

    // making sure the address is never directly accessed
    friend void operator&(non_trivially_relocatable_struct_throwing) = delete;
};

int non_trivially_relocatable_struct_throwing::count = 0;
int non_trivially_relocatable_struct_throwing::move_count = 0;
int non_trivially_relocatable_struct_throwing::dtor_count = 0;

static_assert(
    !is_trivially_relocatable_v<non_trivially_relocatable_struct_throwing>);

int hpx_main()
{
    {
        void* mem1 = std::malloc(N * sizeof(trivially_relocatable_struct));
        void* mem2 = std::malloc(N * sizeof(trivially_relocatable_struct));

        HPX_TEST(mem1 && mem2);

        trivially_relocatable_struct* ptr1 =
            static_cast<trivially_relocatable_struct*>(mem1);
        trivially_relocatable_struct* ptr2 =
            static_cast<trivially_relocatable_struct*>(mem2);

        HPX_TEST(trivially_relocatable_struct::count == 0);
        HPX_TEST(trivially_relocatable_struct::move_count == 0);
        HPX_TEST(trivially_relocatable_struct::dtor_count == 0);

        for (int i = 0; i < N; i++)
        {
            hpx::construct_at(ptr1 + i, 1234);
            hpx::construct_at(ptr2 + i, 5678);
        }

        // N objects constructed
        HPX_TEST(trivially_relocatable_struct::count == 2 * N);

        // relocate M objects to ptr2
        uninitialized_relocate_backward(ptr1, ptr1 + M, ptr2 + N);

        // No move constructor or destructor should be called
        HPX_TEST(trivially_relocatable_struct::move_count == 0);
        HPX_TEST(trivially_relocatable_struct::dtor_count == 0);

        for (int i = M; i < N; i++)
        {
            HPX_TEST(ptr1[i].data == 1234);
        }

        for (int i = 0; i < N - M; i++)
        {
            HPX_TEST(ptr2[i].data == 5678);
        }

        for (int i = N - M; i < N; i++)
        {
            HPX_TEST(ptr2[i].data == 1234);
        }

        std::destroy(ptr1 + M, ptr1 + N);
        std::destroy(ptr2, ptr2 + N);

        std::free(mem1);
        std::free(mem2);
    }
    {
        void* mem1 = std::malloc(N * sizeof(non_trivially_relocatable_struct));
        void* mem2 = std::malloc(N * sizeof(non_trivially_relocatable_struct));

        HPX_TEST(mem1 && mem2);

        non_trivially_relocatable_struct* ptr1 =
            static_cast<non_trivially_relocatable_struct*>(mem1);
        non_trivially_relocatable_struct* ptr2 =
            static_cast<non_trivially_relocatable_struct*>(mem2);

        HPX_TEST(non_trivially_relocatable_struct::count == 0);
        HPX_TEST(non_trivially_relocatable_struct::move_count == 0);
        HPX_TEST(non_trivially_relocatable_struct::dtor_count == 0);

        for (int i = 0; i < N; i++)
        {
            hpx::construct_at(ptr1 + i, 1234);
            hpx::construct_at(ptr2 + i, 5678);
        }

        // N objects constructed
        HPX_TEST(non_trivially_relocatable_struct::count == 2 * N);

        // relocate them to ptr2
        uninitialized_relocate_backward(ptr1, ptr1 + M, ptr2 + N);

        // M move constructors were called and M destructors
        HPX_TEST(non_trivially_relocatable_struct::move_count == M);
        HPX_TEST(non_trivially_relocatable_struct::dtor_count == M);

        for (int i = M; i < N; i++)
        {
            HPX_TEST(ptr1[i].data == 1234);
        }

        for (int i = 0; i < N - M; i++)
        {
            HPX_TEST(ptr2[i].data == 5678);
        }

        for (int i = N - M; i < N; i++)
        {
            HPX_TEST(ptr2[i].data == 1234);
        }

        std::destroy(ptr1 + M, ptr1 + N);
        std::destroy(ptr2, ptr2 + N);

        std::free(mem1);
        std::free(mem2);
    }
    {
        void* mem1 =
            std::malloc(N * sizeof(non_trivially_relocatable_struct_throwing));
        void* mem2 =
            std::malloc(N * sizeof(non_trivially_relocatable_struct_throwing));

        HPX_TEST(mem1 && mem2);

        non_trivially_relocatable_struct_throwing* ptr1 =
            static_cast<non_trivially_relocatable_struct_throwing*>(mem1);
        non_trivially_relocatable_struct_throwing* ptr2 =
            static_cast<non_trivially_relocatable_struct_throwing*>(mem2);

        HPX_TEST(non_trivially_relocatable_struct_throwing::count == 0);
        HPX_TEST(non_trivially_relocatable_struct_throwing::move_count == 0);
        HPX_TEST(non_trivially_relocatable_struct_throwing::dtor_count == 0);

        for (int i = 0; i < N; i++)
        {
            hpx::construct_at(ptr1 + i, 1234);
            hpx::construct_at(ptr2 + i, 5678);
        }

        // N objects constructed
        HPX_TEST(non_trivially_relocatable_struct_throwing::count == 2 * N);

        // relocate them to ptr2
        try
        {
            uninitialized_relocate_backward(ptr1, ptr1 + M, ptr2 + N);
            HPX_UNREACHABLE;    // should have thrown
        }
        catch (...)
        {
        }

        // K move constructors were called, and then the last one throws
        HPX_TEST(non_trivially_relocatable_struct_throwing::move_count == K);

        // K - 1 destructors were called to balance out the move constructors
        // (- 1 because the last move constructor throws)
        // and then M + 1 destructors were called: K on the old range and
        // M - (K - 1) = M - K + 1 on the new range
        HPX_TEST(
            non_trivially_relocatable_struct_throwing::dtor_count == M + K);

        // It stops at K, so K-1 move-destruct pairs have been executed
        // after this N - (K - 1) destructs will be done on the old range
        // and K - 1 on the new range. giving 2*N total destructs

        // The objects in the end of ptr1 are still valid
        for (int i = M; i < N; i++)
        {
            HPX_TEST(ptr1[i].data == 1234);
        }

        // The objects in the beginning of ptr2 are still valid
        for (int i = 0; i < N - M; i++)
        {
            HPX_TEST(ptr2[i].data == 5678);
        }

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
