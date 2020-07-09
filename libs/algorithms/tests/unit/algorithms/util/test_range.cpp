//  Copyright (c) 2015-2017 Francisco Jose Tapia
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/util/range.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <vector>

using namespace hpx::parallel::util;

// template <typename Iter, typename Sent>
// std::ostream& operator<<(std::ostream& out, range<Iter, Sent> R)
// {
//     out << "[ " << (R.end() - R.begin()) << "] ";
//     if (!R.valid())
//         return out;
//     while (R.begin() != R.end())
//         out << (*(R.begin()++)) << " ";
//     out << std::endl;
//     return out;
// }

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

void test1(void)
{
    using iter_t = typename std::vector<std::uint64_t>::iterator;
    using range_t = range<iter_t>;

    std::vector<std::uint64_t> A, B;

    A.resize(10, 0);
    B.resize(10, 0);
    for (uint32_t i = 0; i < 10; ++i)
        A[i] = i;

    range_t RA(A.begin(), A.end()), RB(B.begin(), B.end());

    // test copy copy constructor
    range_t RC(RA);
    HPX_TEST(RC.size() == RA.size());

    RC.begin() = RC.end();
    RC = RA;

    //              test of move
    RB = hpx::parallel::util::init_move(RB, RA);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(B[i] == i);

    //           test of uninitialized_move , destroy
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

    typedef typename std::vector<forensic>::iterator fIter;
    typedef hpx::parallel::util::range<fIter> frange_t;

    char K[160];
    forensic* PAux = reinterpret_cast<forensic*>(&K[0]);
    range<forensic*> F1(PAux, PAux + 20);
    std::vector<forensic> V;
    for (uint32_t i = 0; i < 10; ++i)
        V.emplace_back(i);

    F1 = hpx::parallel::util::uninit_move(F1, frange_t(V.begin(), V.end()));
    for (uint32_t i = 0; i < 10; ++i)
        HPX_TEST(PAux[i].N == i);

    hpx::parallel::util::destroy_range(F1);
}

void test2()
{
    using iter_t = typename std::vector<std::uint64_t>::iterator;

    std::vector<std::uint64_t> V1;
    V1.resize(100, 0);
    range<iter_t> R1(V1.begin(), V1.end());
    uint64_t K = 999;
    range<iter_t> R2 = init(R1, K);
    while (R2.begin() != R2.end())
    {
        HPX_TEST(*R2.begin() == 999);
        R2 = range<iter_t>(R2.begin() + 1, R2.end());
    }
}

// TEST OF HALF_MERGE
void test3()
{
    using iter_t = typename std::vector<std::uint64_t>::iterator;
    using rng = range<iter_t>;
    using compare = std::less<std::uint64_t>;

    compare comp;
    std::vector<std::uint64_t> A, B;
    rng RA(A.begin(), A.end()), RB(B.begin(), B.end());
    rng Rx(A.begin(), A.end());
    rng Rz(Rx);

    // src1 empty
    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(i);

    B.clear();
    RB = rng(B.begin(), B.end());
    HPX_TEST(RB.begin() == RB.end());
    RA = rng(A.begin(), A.end());
    Rx = RA;
    Rz = half_merge(Rx, RB, RA, comp);
    HPX_TEST(Rz.size() == 10);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(*(Rz.begin() + i) == i);

    // src2 empty
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 10; i++)
        B.push_back(i);
    A.resize(10, 0);
    Rz = rng(A.begin(), A.end());
    RB = rng(B.begin(), B.end());
    RA = rng(A.end(), A.end());
    Rx = rng(A.begin(), A.end());
    Rz = half_merge(Rx, RB, RA, comp);
    HPX_TEST(Rz.size() == 10);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(*(Rz.begin() + i) == i);

    // merged even , odd numbers
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 20; i += 2)
        B.push_back(i);
    HPX_TEST(B.size() == 10);
    A.resize(10, 0);
    for (std::uint32_t i = 1; i < 20; i += 2)
        A.push_back(i);
    HPX_TEST(A.size() == 20);
    RA = rng(A.begin() + 10, A.end());
    RB = rng(B.begin(), B.begin() + 10);
    Rx = rng(A.begin(), A.end());
    Rz = half_merge(Rx, RB, RA, comp);
    HPX_TEST(Rz.size() == 20);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(A[i] == i);

    // in src1 0-10 in src2 10-20
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 10; ++i)
        B.push_back(i);
    A.resize(10, 0);
    for (std::uint32_t i = 10; i < 20; i++)
        A.push_back(i);

    RA = rng(A.begin() + 10, A.end());
    RB = rng(B.begin(), B.begin() + 10);
    Rx = rng(A.begin(), A.end());
    Rz = half_merge(Rx, RB, RA, comp);
    HPX_TEST(Rz.size() == 20);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(A[i] == i);

    // in src2 0-10 in src1 10-20
    A.clear();
    B.clear();
    for (std::uint32_t i = 10; i < 20; ++i)
        B.push_back(i);
    A.resize(10, 0);
    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(i);

    RA = rng(A.begin() + 10, A.end());
    RB = rng(B.begin(), B.begin() + 10);
    Rx = rng(A.begin(), A.end());
    Rz = half_merge(Rx, RB, RA, comp);
    HPX_TEST(Rz.size() == 20);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(A[i] == i);
}

// TEST OF STABILITY
void test4()
{
    using iter_t = typename std::vector<xk>::iterator;
    using rng = range<iter_t>;
    using compare = std::less<xk>;

    compare comp;
    std::vector<xk> A, B;
    rng RA(A.begin(), A.end()), RB(B.begin(), B.end());
    rng Rx(A.begin(), A.end());
    rng Rz(Rx);

    // the smallest elements at the beginning of src1
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 50; ++i)
        B.emplace_back(i, 0);
    for (std::uint32_t i = 0; i < 50; ++i)
        B.emplace_back(100 + i * 2, 0);
    A.resize(100, 0);
    for (std::uint32_t i = 0; i < 100; i++)
        A.emplace_back(100 + i * 2, 1);

    RA = rng(A.begin() + 100, A.end());
    RB = rng(B.begin(), B.begin() + 100);
    Rx = rng(A.begin(), A.end());
    Rz = half_merge(Rx, RB, RA, comp);
    HPX_TEST(Rz.size() == 200);

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
    A.resize(100, 0);
    for (std::uint32_t i = 0; i < 50; ++i)
        A.emplace_back(i, 1);
    for (std::uint32_t i = 0; i < 50; ++i)
        A.emplace_back(100 + i * 2, 1);

    for (std::uint32_t i = 0; i < 100; i++)
        B.emplace_back(100 + i * 2, 0);

    RA = rng(A.begin() + 100, A.end());
    RB = rng(B.begin(), B.begin() + 100);
    Rx = rng(A.begin(), A.end());
    Rz = half_merge(Rx, RB, RA, comp);
    HPX_TEST(Rz.size() == 200);

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

// TEST OF FULL_MERGE
void test5()
{
    using compare = std::less<std::uint64_t>;
    using rng = range<std::uint64_t*>;

    std::vector<std::uint64_t> A, B;
    compare comp;

    A.clear();
    B.clear();
    B.assign(21, 0);
    for (std::uint32_t i = 0; i < 20; i += 2)
        A.push_back(i);
    for (std::uint32_t i = 1; i < 20; i += 2)
        A.push_back(i);
    A.push_back(0);

    rng A1(&A[0], &A[10]), A2(&A[10], &A[20]);
    rng B1(&B[0], &B[20]);
    rng C1(A1);

    C1 = full_merge(B1, A1, A2, comp);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(B[i] == i);

    A.clear();
    B.clear();
    B.assign(20, 100);
    for (std::uint32_t i = 0; i < 20; i++)
        A.push_back(i);
    A.push_back(0);

    full_merge(B1, rng(&A[0], &A[10]), rng(&A[10], &A[20]), comp);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(B[i] == i);

    A.clear();
    B.clear();
    B.assign(21, 100);

    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(10 + i);
    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(i);
    A.push_back(0);

    full_merge(B1, rng(&A[0], &A[10]), rng(&A[10], &A[20]), comp);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(B[i] == i);
}

// TEST OF FULL_MERGE
void test6()
{
    typedef typename std::vector<std::uint64_t>::iterator Iter;
    typedef range<Iter> rng;
    typedef std::less<std::uint64_t> compare;

    compare comp;
    std::vector<std::uint64_t> A, B;
    rng RA1(A.begin(), A.end()), RA2(A.begin(), A.end());
    rng RB(B.begin(), B.end());
    rng Rz(RB);

    // src1 empty
    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(i);
    B.clear();
    B.resize(20, 100);

    RB = rng(B.begin(), B.end());
    RA1 = rng(A.begin(), A.begin());
    RA2 = rng(A.begin(), A.end());

    Rz = full_merge(RB, RA1, RA2, comp);
    HPX_TEST(Rz.size() == 10);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(*(Rz.begin() + i) == i);

    // src2 empty
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(i);
    B.resize(10, 0);

    RA1 = rng(A.begin(), A.end());
    RA2 = rng(A.end(), A.end());
    RB = rng(B.begin(), B.end());

    Rz = full_merge(RB, RA1, RA2, comp);
    HPX_TEST(Rz.size() == 10);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(*(Rz.begin() + i) == i);

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

    RA1 = rng(A.begin(), A.begin() + 10);
    RA2 = rng(A.begin() + 10, A.end());
    RB = rng(B.begin(), B.end());

    Rz = full_merge(RB, RA1, RA2, comp);
    HPX_TEST(Rz.size() == 20);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(B[i] == i);

    // in src1 0-10 in src2 10-20
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 20; ++i)
        A.push_back(i);
    B.resize(20, 0);

    RA1 = rng(A.begin(), A.begin() + 10);
    RA2 = rng(A.begin() + 10, A.end());
    RB = rng(B.begin(), B.end());

    Rz = full_merge(RB, RA1, RA2, comp);
    HPX_TEST(Rz.size() == 20);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(B[i] == i);

    // in src2 0-10 in src1 10-20
    A.clear();
    B.clear();
    for (std::uint32_t i = 10; i < 20; ++i)
        A.push_back(i);
    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(i);
    B.resize(20, 0);

    RA1 = rng(A.begin(), A.begin() + 10);
    RA2 = rng(A.begin() + 10, A.end());
    RB = rng(B.begin(), B.end());

    Rz = full_merge(RB, RA1, RA2, comp);
    HPX_TEST(Rz.size() == 20);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(B[i] == i);
};

// TEST OF STABILITY
void test7()
{
    typedef typename std::vector<xk>::iterator Iter;
    typedef range<Iter> rng;
    typedef std::less<xk> compare;

    compare comp;
    std::vector<xk> A, B;
    rng RA1(A.begin(), A.end()), RA2(A.begin(), A.end());
    rng RB(B.begin(), B.end());
    rng Rz(RB);

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

    RA1 = rng(A.begin(), A.begin() + 100);
    RA2 = rng(A.begin() + 100, A.end());
    RB = rng(B.begin(), B.end());

    Rz = full_merge(RB, RA1, RA2, comp);
    HPX_TEST(Rz.size() == 200);

    for (std::uint32_t i = 0; i < 50; ++i)
        HPX_TEST(B[i].num == i && B[i].tail == 0);
    for (std::uint32_t i = 50; i < 150; ++i)
    {
        std::uint32_t K = i + 50;
        std::uint32_t M = K % 2;
        HPX_TEST(B[i].num == K - M && B[i].tail == M);
    }
    for (std::uint32_t i = 150; i < 200; ++i)
        HPX_TEST(B[i].num == 2 * i - 100 && B[i].tail == 1);

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

    RA1 = rng(A.begin(), A.begin() + 100);
    RA2 = rng(A.begin() + 100, A.end());
    RB = rng(B.begin(), B.end());

    Rz = full_merge(RB, RA1, RA2, comp);
    HPX_TEST(Rz.size() == 200);

    for (std::uint32_t i = 0; i < 50; ++i)
        HPX_TEST(B[i].num == i && B[i].tail == 1);
    for (std::uint32_t i = 50; i < 150; ++i)
    {
        std::uint32_t K = i + 50;
        std::uint32_t M = K % 2;
        HPX_TEST(B[i].num == K - M && B[i].tail == M);
    }
    for (std::uint32_t i = 150; i < 200; ++i)
        HPX_TEST(B[i].num == 2 * i - 100 && B[i].tail == 0);
}

// TEST OF UNINITIALIZED_FULL_MERGE
void test8()
{
    typedef std::less<std::uint64_t> compare;
    typedef range<std::uint64_t*> rng;

    std::vector<std::uint64_t> A, B;
    compare comp;

    A.clear();
    B.clear();
    B.assign(21, 0);
    for (std::uint32_t i = 0; i < 20; i += 2)
        A.push_back(i);
    for (std::uint32_t i = 1; i < 20; i += 2)
        A.push_back(i);
    A.push_back(0);

    rng A1(&A[0], &A[10]), A2(&A[10], &A[20]);
    rng B1(&B[0], &B[20]);
    rng C1(A1);

    C1 = uninit_full_merge(B1, A1, A2, comp);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(B[i] == i);

    A.clear();
    B.clear();
    B.assign(21, 100);
    for (std::uint32_t i = 0; i < 20; i++)
        A.push_back(i);
    A.push_back(0);

    uninit_full_merge(B1, rng(&A[0], &A[10]), rng(&A[10], &A[20]), comp);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(B[i] == i);

    A.clear();
    B.clear();
    B.assign(21, 100);

    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(10 + i);
    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(i);
    A.push_back(0);

    uninit_full_merge(B1, rng(&A[0], &A[10]), rng(&A[10], &A[20]), comp);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(B[i] == i);
}

// TEST OF FULL_MERGE
void test9()
{
    typedef typename std::vector<std::uint64_t>::iterator Iter;
    typedef range<Iter> rng;
    typedef std::less<std::uint64_t> compare;

    compare comp;
    std::vector<std::uint64_t> A, B;
    rng RA1(A.begin(), A.end()), RA2(A.begin(), A.end());

    std::uint64_t val = 0;
    range<std::uint64_t*> RB(&val, &val);
    range<std::uint64_t*> Rz(RB);

    // src1 empty
    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(i);
    B.clear();
    B.resize(21, 100);

    RB = range<std::uint64_t*>(&B[0], &B[20]);
    RA1 = rng(A.begin(), A.begin());
    RA2 = rng(A.begin(), A.end());

    Rz = uninit_full_merge(RB, RA1, RA2, comp);
    HPX_TEST(Rz.size() == 10);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(*(Rz.begin() + i) == i);

    // src2 empty
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(i);
    B.resize(11, 0);

    RA1 = rng(A.begin(), A.end());
    RA2 = rng(A.end(), A.end());
    RB = range<std::uint64_t*>(&B[0], &B[10]);

    Rz = uninit_full_merge(RB, RA1, RA2, comp);
    HPX_TEST(Rz.size() == 10);
    for (std::uint32_t i = 0; i < 10; ++i)
        HPX_TEST(*(Rz.begin() + i) == i);

    // merged even , odd numbers
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 20; i += 2)
        A.push_back(i);
    HPX_TEST(A.size() == 10);
    for (std::uint32_t i = 1; i < 20; i += 2)
        A.push_back(i);
    HPX_TEST(A.size() == 20);
    B.resize(21, 0);

    RA1 = rng(A.begin(), A.begin() + 10);
    RA2 = rng(A.begin() + 10, A.end());
    RB = range<std::uint64_t*>(&B[0], &B[20]);

    Rz = uninit_full_merge(RB, RA1, RA2, comp);
    HPX_TEST(Rz.size() == 20);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(B[i] == i);

    // in src1 0-10 in src2 10-20
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 20; ++i)
        A.push_back(i);
    B.resize(21, 0);

    RA1 = rng(A.begin(), A.begin() + 10);
    RA2 = rng(A.begin() + 10, A.end());
    RB = range<std::uint64_t*>(&B[0], &B[20]);

    Rz = uninit_full_merge(RB, RA1, RA2, comp);
    HPX_TEST(Rz.size() == 20);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(B[i] == i);

    // in src2 0-10 in src1 10-20
    A.clear();
    B.clear();
    for (std::uint32_t i = 10; i < 20; ++i)
        A.push_back(i);
    for (std::uint32_t i = 0; i < 10; i++)
        A.push_back(i);
    B.resize(21, 0);

    RA1 = rng(A.begin(), A.begin() + 10);
    RA2 = rng(A.begin() + 10, A.end());
    RB = range<std::uint64_t*>(&B[0], &B[20]);

    Rz = uninit_full_merge(RB, RA1, RA2, comp);
    HPX_TEST(Rz.size() == 20);
    for (std::uint32_t i = 0; i < 20; ++i)
        HPX_TEST(B[i] == i);
}

// TEST OF STABILITY
void test10()
{
    typedef typename std::vector<xk>::iterator Iter;
    typedef range<Iter> rng;
    typedef std::less<xk> compare;

    compare comp;
    std::vector<xk> A, B;
    rng RA1(A.begin(), A.end()), RA2(A.begin(), A.end());
    range<xk*> RB;
    range<xk*> Rz;

    // the smallest elements at the beginning of src1
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 50; ++i)
        A.emplace_back(i, 0);
    for (std::uint32_t i = 0; i < 50; ++i)
        A.emplace_back(100 + i * 2, 0);
    for (std::uint32_t i = 0; i < 100; i++)
        A.emplace_back(100 + i * 2, 1);
    B.resize(201, 0);

    RA1 = rng(A.begin(), A.begin() + 100);
    RA2 = rng(A.begin() + 100, A.end());
    RB = range<xk*>(&B[0], &B[200]);

    Rz = uninit_full_merge(RB, RA1, RA2, comp);
    HPX_TEST(Rz.size() == 200);

    for (std::uint32_t i = 0; i < 50; ++i)
        HPX_TEST(B[i].num == i && B[i].tail == 0);
    for (std::uint32_t i = 50; i < 150; ++i)
    {
        std::uint32_t K = i + 50;
        std::uint32_t M = K % 2;
        HPX_TEST(B[i].num == K - M && B[i].tail == M);
    };
    for (std::uint32_t i = 150; i < 200; ++i)
        HPX_TEST(B[i].num == 2 * i - 100 && B[i].tail == 1);

    // the smallest elements at the beginning of src2
    A.clear();
    B.clear();
    for (std::uint32_t i = 0; i < 100; i++)
        A.emplace_back(100 + i * 2, 0);
    for (std::uint32_t i = 0; i < 50; ++i)
        A.emplace_back(i, 1);
    for (std::uint32_t i = 0; i < 50; ++i)
        A.emplace_back(100 + i * 2, 1);
    B.resize(201, 0);

    RA1 = rng(A.begin(), A.begin() + 100);
    RA2 = rng(A.begin() + 100, A.end());
    RB = range<xk*>(&B[0], &B[200]);

    Rz = uninit_full_merge(RB, RA1, RA2, comp);
    HPX_TEST(Rz.size() == 200);

    for (std::uint32_t i = 0; i < 50; ++i)
        HPX_TEST(B[i].num == i && B[i].tail == 1);
    for (std::uint32_t i = 50; i < 150; ++i)
    {
        std::uint32_t K = i + 50;
        std::uint32_t M = K % 2;
        HPX_TEST(B[i].num == K - M && B[i].tail == M);
    };
    for (std::uint32_t i = 150; i < 200; ++i)
        HPX_TEST(B[i].num == 2 * i - 100 && B[i].tail == 0);
}

int main(int, char*[])
{
    test1();
    test2();
    test3();
    test4();
    test5();
    test6();
    test7();
    test8();
    test9();
    test10();

    return hpx::util::report_errors();
};
