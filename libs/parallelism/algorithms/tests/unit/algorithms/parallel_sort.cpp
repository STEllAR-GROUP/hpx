//  Copyright (c) 2015-2019 Francisco Jose Tapia
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/algorithms/sort.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#if defined(HPX_DEBUG)
#define NUMELEMS 50000
#else
#define NUMELEMS 5000000
#endif

void test1()
{
    typedef std::less<std::uint64_t> compare;

    constexpr std::uint32_t NElem = NUMELEMS;
    std::vector<std::uint64_t> V1, V2;
    std::mt19937_64 my_rand(std::rand());
    compare comp;

    for (std::uint32_t i = 0; i < NElem; ++i)
        V1.push_back(my_rand() % NElem);
    V2 = V1;

    hpx::parallel::sort(hpx::execution::par, V1.begin(), V1.end(), comp);

    for (unsigned i = 1; i < NElem; i++)
    {
        HPX_TEST(V1[i - 1] <= V1[i]);
    }

    std::sort(V2.begin(), V2.end(), comp);
    for (unsigned i = 1; i < NElem; i++)
    {
        HPX_TEST(V1[i] == V2[i]);
    }

    V1.clear();
    for (std::uint32_t i = 0; i < NElem; ++i)
        V1.push_back(i);

    hpx::parallel::sort(hpx::execution::par, V1.begin(), V1.end(), comp);

    for (unsigned i = 1; i < NElem; i++)
    {
        HPX_TEST(V1[i] == i);
    }

    V1.clear();
    for (std::uint32_t i = 0; i < NElem; ++i)
        V1.push_back(NElem - i);

    hpx::parallel::sort(hpx::execution::par, V1.begin(), V1.end(), comp);
    for (unsigned i = 1; i < NElem; i++)
    {
        HPX_TEST(V1[i] == (i + 1));
    }

    V1.clear();
    for (std::uint32_t i = 0; i < NElem; ++i)
        V1.push_back(1000);

    hpx::parallel::sort(hpx::execution::par, V1.begin(), V1.end(), comp);
    for (unsigned i = 1; i < NElem; i++)
    {
        HPX_TEST(V1[i] == 1000);
    }
}

void test2()
{
    typedef typename std::vector<std::uint64_t>::iterator iter_t;

#if defined(HPX_DEBUG)
    constexpr std::uint32_t NELEM = 41667;
#else
    constexpr std::uint32_t NELEM = 416667;
#endif

    std::vector<std::uint64_t> A;
    std::less<std::uint64_t> comp;

    for (std::uint32_t i = 0; i < 1000; ++i)
        A.push_back(0);
    for (std::uint32_t i = 0; i < NELEM; ++i)
        A.push_back(NELEM - i);
    for (std::uint32_t i = 0; i < 1000; ++i)
        A.push_back(0);

    hpx::parallel::sort(hpx::execution::par, A.begin() + 1000,
        A.begin() + (1000 + NELEM), comp);
    for (iter_t it = A.begin() + 1000; it != A.begin() + (1000 + NELEM); ++it)
    {
        HPX_TEST((*(it - 1)) <= (*it));
    }

    HPX_TEST(A[998] == 0 && A[999] == 0 && A[1000 + NELEM] == 0 &&
        A[1001 + NELEM] == 0);

    A.clear();
    for (std::uint32_t i = 0; i < 1000; ++i)
        A.push_back(999999999);
    for (std::uint32_t i = 0; i < NELEM; ++i)
        A.push_back(NELEM - i);
    for (std::uint32_t i = 0; i < 1000; ++i)
        A.push_back(999999999);

    hpx::parallel::sort(hpx::execution::par, A.begin() + 1000,
        A.begin() + (1000 + NELEM), comp);

    for (iter_t it = A.begin() + 1001; it != A.begin() + (1000 + NELEM); ++it)
    {
        HPX_TEST((*(it - 1)) <= (*it));
    }
    HPX_TEST(A[998] == 999999999 && A[999] == 999999999 &&
        A[1000 + NELEM] == 999999999 && A[1001 + NELEM] == 999999999);
}

void test3()
{
    typedef std::less<std::uint32_t> compare;

    constexpr std::uint32_t NElem = NUMELEMS;
    std::vector<std::uint32_t> V1, V2;
    V1.reserve(NElem);
    std::mt19937 my_rand(std::rand());

    for (std::uint32_t i = 0; i < NElem; ++i)
        V1.push_back(my_rand());
    V2 = V1;

    std::sort(V2.begin(), V2.end());

    hpx::parallel::sort(hpx::execution::par, V1.begin(), V1.end(), compare());

    for (unsigned i = 0; i < V1.size(); i++)
    {
        HPX_TEST(V1[i] == V2[i]);
    }
}

template <std::uint32_t NN>
struct int_array
{
    std::uint64_t M[NN];

    int_array(std::uint64_t number = 0)
    {
        for (std::uint32_t i = 0; i < NN; ++i)
            M[i] = number;
    }

    bool operator<(const int_array<NN>& A) const
    {
        return M[0] < A.M[0];
    }

    bool operator==(const int_array<NN>& A) const
    {
        bool sw = true;
        for (std::uint32_t i = 0; i < NN; ++i)
            sw &= (M[i] == A.M[i]);
        return sw;
    }
};

void test4()
{
    std::less<std::uint64_t> cmp64;
    std::less<std::uint32_t> cmp32;
    std::less<std::uint16_t> cmp16;
    std::less<std::uint8_t> cmp8;

    std::mt19937_64 my_rand(std::rand());

#if defined(HPX_DEBUG)
    constexpr std::uint32_t NELEM = (1 << 18);
#else
    constexpr std::uint32_t NELEM = (1 << 24);
#endif

    std::vector<std::uint64_t> V1, V2, V3;
    V1.reserve(NELEM);
    V2.reserve(NELEM);

    for (std::uint32_t i = 0; i < NELEM; ++i)
        V1.push_back(my_rand());
    V3 = V2 = V1;

    // 64 bits elements
    std::uint64_t* p64_1 = &V1[0];
    hpx::parallel::sort(hpx::execution::par, p64_1, p64_1 + NELEM, cmp64);

    for (unsigned i = 1; i < NELEM; i++)
    {
        HPX_TEST(p64_1[i - 1] <= p64_1[i]);
    }

    std::uint64_t* p64_2 = &V2[0];
    std::sort(p64_2, p64_2 + NELEM, cmp64);
    for (unsigned i = 0; i < NELEM; i++)
    {
        HPX_TEST(p64_1[i] <= p64_2[i]);
    }

    // 32 bits elements
    V1 = V2 = V3;
    std::uint32_t* p32_1 = reinterpret_cast<std::uint32_t*>(&V1[0]);
    hpx::parallel::sort(
        hpx::execution::par, p32_1, p32_1 + (NELEM << 1), cmp32);

    for (unsigned i = 1; i < (NELEM << 1); i++)
    {
        HPX_TEST(p32_1[i - 1] <= p32_1[i]);
    }

    std::uint32_t* p32_2 = reinterpret_cast<std::uint32_t*>(&V2[0]);
    std::sort(p32_2, p32_2 + (NELEM << 1), cmp32);
    for (unsigned i = 0; i < (NELEM << 1); i++)
    {
        HPX_TEST(p32_1[i] == p32_2[i]);
    }

    // 16 bits elements
    V1 = V2 = V3;
    std::uint16_t* p16_1 = reinterpret_cast<std::uint16_t*>(&V1[0]);
    hpx::parallel::sort(
        hpx::execution::par, p16_1, p16_1 + (NELEM << 2), cmp16);
    for (unsigned i = 1; i < (NELEM << 2); i++)
    {
        HPX_TEST(p16_1[i - 1] <= p16_1[i]);
    }

    std::uint16_t* p16_2 = reinterpret_cast<std::uint16_t*>(&V2[0]);
    std::sort(p16_2, p16_2 + (NELEM << 2), cmp16);
    for (unsigned i = 1; i < (NELEM << 2); i++)
    {
        HPX_TEST(p16_1[i] == p16_2[i]);
    };

    // 8 bits elements
    V1 = V2 = V3;
    std::uint8_t* p8_1 = reinterpret_cast<std::uint8_t*>(&V1[0]);
    hpx::parallel::sort(hpx::execution::par, p8_1, p8_1 + (NELEM << 3), cmp8);
    for (unsigned i = 1; i < (NELEM << 3); i++)
    {
        HPX_TEST(p8_1[i - 1] <= p8_1[i]);
    }

    std::uint8_t* p8_2 = reinterpret_cast<std::uint8_t*>(&V2[0]);
    std::sort(p8_2, p8_2 + (NELEM << 3), cmp8);
    for (unsigned i = 1; i < (NELEM << 3); i++)
    {
        HPX_TEST(p8_1[i] == p8_2[i]);
    }
}

template <typename IA>
void test_int_array(std::uint32_t NELEM)
{
    typedef std::less<IA> compare;
    std::mt19937_64 my_rand(std::rand());

    std::vector<IA> V1, V2;
    V1.reserve(NELEM);
    for (std::uint32_t i = 0; i < NELEM; ++i)
        V1.emplace_back(my_rand());
    V2 = V1;
    hpx::parallel::sort(hpx::execution::par, V1.begin(), V1.end(), compare());
    for (unsigned i = 1; i < NELEM; i++)
    {
        HPX_TEST(!(V1[i] < V1[i - 1]));
    }
    std::sort(V2.begin(), V2.end(), compare());
    for (unsigned i = 1; i < NELEM; i++)
    {
        HPX_TEST(V1[i] == V2[i]);
    }
}

void test5()
{
#if defined(HPX_DEBUG)
#define NUMELEMS_SHIFT 13
#else
#define NUMELEMS_SHIFT 17
#endif

    test_int_array<int_array<1>>(1u << (NUMELEMS_SHIFT + 3));
    test_int_array<int_array<2>>(1u << (NUMELEMS_SHIFT + 2));
    test_int_array<int_array<4>>(1u << (NUMELEMS_SHIFT + 1));
    test_int_array<int_array<8>>(1u << NUMELEMS_SHIFT);
    test_int_array<int_array<16>>(1u << NUMELEMS_SHIFT);
    test_int_array<int_array<32>>(1u << NUMELEMS_SHIFT);
    test_int_array<int_array<64>>(1u << NUMELEMS_SHIFT);
    test_int_array<int_array<128>>(1u << NUMELEMS_SHIFT);
}

void test6()
{
    std::mt19937_64 my_rand(std::rand());
    constexpr std::uint32_t NELEM = 1 << 20;
    constexpr std::uint32_t NString = 100000;

    std::vector<std::uint64_t> V1;
    V1.reserve(NELEM);

    for (std::uint32_t i = 0; i < NELEM; ++i)
        V1.push_back(my_rand());

    std::uint64_t* p64 = &V1[0];
    char* pchar = reinterpret_cast<char*>(p64);

    std::string sinput(pchar, (NELEM << 3));

    std::istringstream strm_input(sinput);
    std::string inval;

    std::vector<std::string> V, VAux;
    V.reserve(NString);

    strm_input.seekg(0, std::ios_base::beg);
    strm_input.seekg(0, std::ios_base::beg);

    for (size_t i = 0; i < NString; ++i)
    {
        if (!strm_input.eof())
        {
            strm_input >> inval;
            V.push_back(inval);
            inval.clear();
        }
        else
        {
            throw std::ios_base::failure("Insufficient length of the vector\n");
        }
    }

    VAux = V;
    typedef std::less<std::string> compare;
    hpx::parallel::sort(hpx::execution::par, V.begin(), V.end(), compare());

    for (unsigned i = 1; i < NString; i++)
    {
        HPX_TEST(!(V[i] < V[i - 1]));
    }

    std::sort(VAux.begin(), VAux.end(), compare());
    for (unsigned i = 1; i < NString; i++)
    {
        HPX_TEST(V[i] == VAux[i]);
    }
}

int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test1();
    test2();
    test3();
    test4();
    test5();
    test6();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run");

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
