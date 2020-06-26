//  Copyright (c) 2015-2017 Francisco Jose Tapia
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/util/low_level.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace hpx::parallel::util;

struct xk
{
    unsigned tail : 3;
    unsigned num : 24;

    xk(unsigned N = 0, unsigned T = 0)
      : tail(T)
      , num(N)
    {
    }

    bool operator<(xk A) const
    {
        return (unsigned) num < (unsigned) A.num;
    }
};

// std::ostream& operator<<(std::ostream& out, xk x)
// {
//     out << "[" << x.num << "-" << x.tail << "] ";
//     return out;
// }

//  TEST MOVE, CONSTRUCT, UNINITIALIZED_MOVE, DESTROY
void test1()
{
    std::vector<std::uint64_t> A, B;

    A.resize(10, 0);
    B.resize(10, 0);
    for (std::uint32_t i = 0; i < 10; ++i)
        A[i] = i;

    init_move(&B[0], A.begin(), A.end());
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(B[i] == i);

    // test of construct , destroy && uninitialized_move
    struct forensic
    {
        std::int64_t N;
        forensic(std::uint64_t K = 0)
        {
            N = (std::int64_t) K;
        }

        ~forensic()
        {
            N = -1;
        }
    };

    char K[80];
    forensic* PAux = reinterpret_cast<forensic*>(&K[0]);

    for (std::uint32_t i = 0; i < 10; ++i)
        construct(PAux + i, i);
    for (std::uint32_t i = 0; i < 10; ++i)
    {
        HPX_TEST(PAux[i].N == i);
    }
    destroy(PAux, PAux + 10);

    // test of uninitialized_move
    std::vector<forensic> V;
    for (std::uint32_t i = 0; i < 10; ++i)
        V.emplace_back(i);

    uninit_move(PAux, V.begin(), V.end());
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(PAux[i].N == i);

    destroy(PAux, PAux + 10);
}

// TEST OF FULL_MERGE
void test2()
{
    typedef std::less<std::uint64_t> compare;
    std::vector<std::uint64_t> A, B;
    size_t NA = 0;

    A.clear();
    B.clear();
    B.assign(21, 0);
    for (std::uint32_t i = 0; i < 20; i += 2)
        A.push_back(i);
    for (std::uint32_t i = 1; i < 20; i += 2)
        A.push_back(i);
    A.push_back(0);

    full_merge(&A[0], &A[10], &A[10], &A[20], &B[0], compare());
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(B[i] == i);

    for (std::uint32_t i = 0; i < 20; ++i)
        B[i] = 100;
    for (std::uint32_t i = 0; i < 20; i++)
        A[i] = i;

    full_merge(&A[0], &A[10], &A[10], &A[20], &B[0], compare());
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(B[i] == i);

    NA = 0;
    for (std::uint32_t i = 0; i < 20; ++i)
        B[i] = 100;
    for (std::uint32_t i = 0; i < 10; i++)
        A[NA++] = 10 + i;
    for (std::uint32_t i = 0; i < 10; i++)
        A[NA++] = i;
    full_merge(&A[0], &A[10], &A[10], &A[20], &B[0], compare());
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(B[i] == i);
}

// TEST OF HALF_MERGE
void test3()
{
    typedef std::less<std::uint64_t> compare;
    std::vector<std::uint64_t> A, B;

    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 20; i += 2)
        B.push_back(i);
    B.push_back(0);
    A.resize(10, 0);
    for (std::uint32_t i = 1; i < 20; i += 2)
        A.push_back(i);
    A.push_back(0);
    half_merge(&B[0], &B[10], &A[10], &A[20], &A[0], compare());
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(A[i] == i);

    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 10; ++i)
        B.push_back(i);
    B.push_back(0);
    A.resize(10, 0);
    for (std::uint32_t i = 10; i < 20; i++)
        A.push_back(i);
    A.push_back(0);
    half_merge(&B[0], &B[10], &A[10], &A[20], &A[0], compare());
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(A[i] == i);

    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 10; i++)
        B.push_back(10 + i);
    B.push_back(0);
    A.resize(10, 0);
    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(i);
    A.push_back(0);
    half_merge(&B[0], &B[10], &A[10], &A[20], &A[0], compare());
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(A[i] == i);
}

// TEST OF UNINITIALIZED_FULL_MERGE
void test5()
{
    struct forensic
    {
        std::int64_t N;
        forensic(std::uint64_t K = 0)
        {
            N = (std::int64_t) K;
        }

        ~forensic()
        {
            N = -1;
        }

        bool operator<(const forensic& f) const
        {
            return (N < f.N);
        }
    };

    char K[1600];
    forensic* PAux = reinterpret_cast<forensic*>(&K[0]);

    typedef std::less<forensic> compare;
    std::vector<forensic> A;

    A.clear();
    for (std::uint32_t i = 0; i < 20; i += 2)
        A.emplace_back(i);
    for (std::uint32_t i = 1; i < 20; i += 2)
        A.emplace_back(i);
    A.emplace_back(0);

    uninit_full_merge(&A[0], &A[10], &A[10], &A[20], PAux, compare());
    for (std::uint32_t i = 0; i < 20; ++i)
    {
        HPX_TEST(PAux[i].N == i);
    }
    destroy(PAux, PAux + 20);

    for (std::uint32_t i = 0; i < 20; i++)
        A[i] = i;

    uninit_full_merge(&A[0], &A[10], &A[10], &A[20], PAux, compare());
    for (std::uint32_t i = 0; i < 20; ++i)
    {
        HPX_TEST(PAux[i].N == i);
    }
    destroy(PAux, PAux + 20);

    for (std::uint32_t i = 0; i < 10; i++)
        A[i] = 10 + i;
    for (std::uint32_t i = 0; i < 10; i++)
        A[10 + i] = i;
    uninit_full_merge(&A[0], &A[10], &A[10], &A[20], PAux, compare());
    for (std::uint32_t i = 0; i < 20; ++i)
    {
        HPX_TEST(PAux[i].N == i);
    }
    destroy(PAux, PAux + 20);
}

// TEST OF in_place_MERGE
void test6()
{
    typedef std::less<std::uint64_t> compare;
    std::vector<std::uint64_t> A, B;
    compare comp;

    A.clear();
    B.clear();
    B.assign(20, 0);
    for (std::uint32_t i = 0; i < 20; i += 2)
        A.push_back(i);
    for (std::uint32_t i = 1; i < 20; i += 2)
        A.push_back(i);
    A.push_back(0);

    bool SW = in_place_merge(&A[0], &A[10], &A[20], &B[0], comp);
    HPX_TEST(SW == false);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(A[i] == i);

    A.clear();
    B.clear();
    B.assign(20, 100);
    for (std::uint32_t i = 0; i < 20; i++)
        A.push_back(i);
    A.push_back(0);

    in_place_merge(&A[0], &A[10], &A[20], &B[0], comp);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(A[i] == i);

    A.clear();
    B.clear();
    B.assign(20, 100);

    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(10 + i);
    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(i);
    A.push_back(0);

    in_place_merge(&A[0], &A[10], &A[20], &B[0], comp);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(A[i] == i);
};

// TEST OF in_place_MERGE
void test7()
{
    typedef std::less<std::uint64_t> compare;
    compare comp;
    std::vector<std::uint64_t> A, B;

    // src1 empty
    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(i);
    B.clear();
    B.resize(20, 100);

    in_place_merge(A.begin(), A.begin(), A.end(), B.begin(), comp);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(A[i] == i);

    // src2 empty
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(i);
    B.resize(10, 0);

    in_place_merge(A.begin(), A.end(), A.end(), B.begin(), comp);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(A[i] == i);

    // merged even , odd numbers
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 20; i += 2)
        A.push_back(i);
    HPX_TEST(A.size() == 10);
    for (std::uint32_t i = 1; i < 20; i += 2)
        A.push_back(i);
    HPX_TEST(A.size() == 20);
    B.resize(20, 0);

    in_place_merge(A.begin(), A.begin() + 10, A.end(), B.begin(), comp);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(A[i] == i);

    // in src1 0-10 in src2 10-20
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 20; ++i)
        A.push_back(i);
    B.resize(20, 0);

    in_place_merge(A.begin(), A.begin() + 10, A.end(), B.begin(), comp);
    HPX_TEST(A.size() == 20);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(A[i] == i);

    // in src2 0-10 in src1 10-20
    A.clear();
    B.clear();
    for (std::uint32_t i = 10; i < 20; ++i)
        A.push_back(i);
    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(i);
    B.resize(20, 0);

    in_place_merge(A.begin(), A.begin() + 10, A.end(), B.begin(), comp);
    HPX_TEST(A.size() == 20);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(A[i] == i);
}

// TEST OF STABILITY
void test8()
{
    typedef std::less<xk> compare;
    compare comp;
    std::vector<xk> A, B;

    // the smallest elements at the beginning of src1
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 50; ++i)
        A.emplace_back(i, 0);
    for (std::uint32_t i = 0; i < 50; ++i)
        A.emplace_back(100 + i * 2, 0);
    for (std::uint32_t i = 0; i < 100; i++)
        A.emplace_back(100 + i * 2, 1);
    B.resize(200, 0);

    in_place_merge(A.begin(), A.begin() + 100, A.end(), B.begin(), comp);
    for (std::uint32_t i = 0; i < 50; ++i)
        HPX_TEST(A[i].num == i && A[i].tail == 0);
    for (std::uint32_t i = 50; i < 150; ++i)
    {
        std::uint32_t K = i + 50;
        std::uint32_t M = K % 2;
        HPX_TEST(A[i].num == K - M && A[i].tail == M);
    }
    for (std::uint32_t i = 150; i < 200; ++i)
        HPX_TEST(A[i].num == 2 * i - 100 && A[i].tail == 1);

    // the smallest elements at the beginning of src2
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 100; i++)
        A.emplace_back(100 + i * 2, 0);
    for (std::uint32_t i = 0; i < 50; ++i)
        A.emplace_back(i, 1);
    for (std::uint32_t i = 0; i < 50; ++i)
        A.emplace_back(100 + i * 2, 1);
    B.resize(200, 0);

    in_place_merge(A.begin(), A.begin() + 100, A.end(), B.begin(), comp);

    for (std::uint32_t i = 0; i < 50; ++i)
        HPX_TEST(A[i].num == i && A[i].tail == 1);
    for (std::uint32_t i = 50; i < 150; ++i)
    {
        std::uint32_t K = i + 50;
        std::uint32_t M = K % 2;
        HPX_TEST(A[i].num == K - M && A[i].tail == M);
    }
    for (std::uint32_t i = 150; i < 200; ++i)
        HPX_TEST(A[i].num == 2 * i - 100 && A[i].tail == 0);
}

// TEST OF IN_PLACE_MERGE_UNCONTIGUOUS
void test9()
{
    typedef std::less<std::uint64_t> compare;
    bool SW;

    std::vector<std::uint64_t> A, B, C;
    compare comp;

    A.clear();
    B.clear();
    C.clear();
    C.assign(10, 0);
    for (std::uint32_t i = 0; i < 20; i += 2)
        A.push_back(i);
    A.push_back(0);
    for (std::uint32_t i = 1; i < 20; i += 2)
        B.push_back(i);
    B.push_back(0);

    SW = in_place_merge_uncontiguous(&A[0], &A[10], &B[0], &B[10], &C[0], comp);
    HPX_TEST(SW == false);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(A[i] == i);
    for (std::uint32_t i = 10; i < 20; ++i)
        HPX_TEST(B[i - 10] == i);

    A.clear();
    B.clear();
    C.clear();
    C.assign(10, 0);
    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(i);
    A.push_back(0);
    for (std::uint32_t i = 0; i < 10; i++)
        B.push_back(i + 10);
    B.push_back(0);

    SW = in_place_merge_uncontiguous(&A[0], &A[10], &B[0], &B[10], &C[0], comp);
    HPX_TEST(SW == true);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(A[i] == i);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(B[i] == i + 10);

    A.clear();
    B.clear();
    C.clear();
    C.assign(10, 0);
    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(10 + i);
    A.push_back(0);
    for (std::uint32_t i = 0; i < 10; i++)
        B.push_back(i);
    B.push_back(0);

    SW = in_place_merge_uncontiguous(&A[0], &A[10], &B[0], &B[10], &C[0], comp);
    HPX_TEST(SW == false);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(A[i] == i);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(B[i] == i + 10);
}

// TEST OF in_place_MERGE
void test10()
{
    typedef std::less<std::uint64_t> compare;
    compare comp;
    std::vector<std::uint64_t> A, B, C;
    bool SW;

    // src1 empty
    A.clear();
    B.clear();
    C.clear();

    for (std::uint32_t i = 0; i < 10; i++)
        B.push_back(i);
    C.resize(10, 0);
    SW = in_place_merge_uncontiguous(
        A.begin(), A.end(), B.begin(), B.end(), C.begin(), comp);
    HPX_TEST(SW == true);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(B[i] == i);

    // src2 empty
    A.clear();
    B.clear();
    C.resize(10, 0);
    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(i);
    SW = in_place_merge_uncontiguous(
        A.begin(), A.end(), B.begin(), B.end(), C.begin(), comp);
    HPX_TEST(SW == true);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(A[i] == i);

    // merged even , odd numbers
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 20; i += 2)
        A.push_back(i);
    HPX_TEST(A.size() == 10);
    for (std::uint32_t i = 1; i < 20; i += 2)
        B.push_back(i);
    HPX_TEST(B.size() == 10);
    SW = in_place_merge_uncontiguous(
        A.begin(), A.end(), B.begin(), B.end(), C.begin(), comp);
    HPX_TEST(SW == false);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(A[i] == i);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(B[i] == i + 10);

    // in src1 0-10 in src2 10-20
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 10; ++i)
        A.push_back(i);
    for (std::uint32_t i = 10; i < 20; ++i)
        B.push_back(i);
    SW = in_place_merge_uncontiguous(
        A.begin(), A.end(), B.begin(), B.end(), C.begin(), comp);
    HPX_TEST(SW == true);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(A[i] == i);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(B[i] == i + 10);

    // in src2 0-10 in src1 10-20
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 10; ++i)
        B.push_back(i);
    for (std::uint32_t i = 10; i < 20; i++)
        A.push_back(i);
    SW = in_place_merge_uncontiguous(
        A.begin(), A.end(), B.begin(), B.end(), C.begin(), comp);
    HPX_TEST(SW == false);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(A[i] == i);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(B[i] == i + 10);
}

// TEST OF STABILITY
void test11()
{
    typedef std::less<xk> compare;
    compare comp;
    std::vector<xk> A, B, C;
    bool SW;

    //------------------------------------------------------------------------
    // the smallest elements at the beginning of src1
    //------------------------------------------------------------------------
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 50; ++i)
        A.emplace_back(i, 0);
    for (std::uint32_t i = 0; i < 50; ++i)
        A.emplace_back(100 + i * 2, 0);
    for (std::uint32_t i = 0; i < 100; i++)
        B.emplace_back(100 + i * 2, 1);
    C.resize(100, 0);
    SW = in_place_merge_uncontiguous(
        A.begin(), A.end(), B.begin(), B.end(), C.begin(), comp);
    HPX_TEST(SW == false);
    for (std::uint32_t i = 0; i < 50; ++i)
        HPX_TEST(A[i].num == i && A[i].tail == 0);
    for (std::uint32_t i = 50; i < 100; ++i)
    {
        std::uint32_t K = i + 50;
        std::uint32_t M = K % 2;
        HPX_TEST(A[i].num == K - M && A[i].tail == M);
    }

    for (std::uint32_t i = 0; i < 50; ++i)
    {
        std::uint32_t K = i + 150;
        std::uint32_t M = K % 2;
        HPX_TEST(B[i].num == K - M && B[i].tail == M);
    }

    for (std::uint32_t i = 50; i < 100; ++i)
        HPX_TEST(B[i].num == 2 * i + 100 && B[i].tail == 1);

    // the smallest elements at the beginning of src2
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 100; i++)
        A.emplace_back(100 + i * 2, 0);
    for (std::uint32_t i = 0; i < 50; ++i)
        B.emplace_back(i, 1);
    for (std::uint32_t i = 0; i < 50; ++i)
        B.emplace_back(100 + i * 2, 1);

    SW = in_place_merge_uncontiguous(
        A.begin(), A.end(), B.begin(), B.end(), C.begin(), comp);
    HPX_TEST(SW == false);

    for (std::uint32_t i = 0; i < 50; ++i)
        HPX_TEST(A[i].num == i && A[i].tail == 1);
    for (std::uint32_t i = 50; i < 100; ++i)
    {
        std::uint32_t K = i + 50;
        std::uint32_t M = K % 2;
        HPX_TEST(A[i].num == K - M && A[i].tail == M);
    }

    for (std::uint32_t i = 0; i < 50; ++i)
    {
        std::uint32_t K = i + 150;
        std::uint32_t M = K % 2;
        HPX_TEST(B[i].num == K - M && B[i].tail == M);
    }

    for (std::uint32_t i = 50; i < 100; ++i)
        HPX_TEST(B[i].num == 2 * i + 100 && B[i].tail == 0);
}

void test12()
{
    std::vector<std::uint32_t> A = {2, 3, 4, 6, 10, 11, 12, 13};
    std::vector<std::uint32_t> B = {5, 7, 8, 9}, C(8, 0);

    bool SW = in_place_merge_uncontiguous(A.begin(), A.end(), B.begin(),
        B.end(), C.begin(), std::less<std::uint32_t>());
    HPX_TEST(SW == false);
    for (std::uint32_t i = 0; i < A.size(); ++i)
        HPX_TEST(A[i] == i + 2);
    for (std::uint32_t i = 0; i < B.size(); ++i)
        HPX_TEST(B[i] == i + 10);
}

int main(int, char*[])
{
    test1();
    test2();
    test3();
    test5();
    test6();
    test7();
    test8();
    test9();
    test10();
    test11();
    test12();

    return hpx::util::report_errors();
}
