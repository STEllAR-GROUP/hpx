//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This benchmark provides an equivalent for the benchmarks published at
// https://github.com/atemerev/skynet. It is called the Skynet 1M concurrency
// micro benchmark.
//
// It creates an actor (goroutine, whatever), which spawns 10 new actors, each
// of them spawns 10 more actors, etc. until one million actors are created on
// the final level. Then, each of them returns back its ordinal number (from 0
// to 999999), which are summed on the previous level and sent back upstream,
// until reaching the root actor. (The answer should be 499999500000).

// This code implements two versions of the skynet micro benchmark: a 'normal'
// and a futurized one.

#include <hpx/hpx_main.hpp>
#include <hpx/hpx.hpp>
#include <hpx/iostream.hpp>

#include <cstdint>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
std::int64_t skynet(std::int64_t num, std::int64_t size, std::int64_t div)
{
    if (size != 1)
    {
        size /= div;

        std::vector<hpx::future<std::int64_t> > results;
        results.reserve(div);

        for (std::int64_t i = 0; i != div; ++i)
        {
            std::int64_t sub_num = num + i * size;
            results.push_back(hpx::async(skynet, sub_num, size, div));
        }

        hpx::wait_all(results);

        std::int64_t sum = 0;
        for (auto & f : results)
            sum += f.get();
        return sum;
    }
    return num;
}

///////////////////////////////////////////////////////////////////////////////
hpx::future<std::int64_t>
skynet_f(std::int64_t num, std::int64_t size, std::int64_t div)
{
    if (size != 1)
    {
        size /= div;

        std::vector<hpx::future<std::int64_t> > results;
        results.reserve(div);

        for (std::int64_t i = 0; i != div; ++i)
        {
            std::int64_t sub_num = num + i * size;
            results.push_back(hpx::async(skynet_f, sub_num, size, div));
        }

        return hpx::dataflow(
            [](std::vector<hpx::future<std::int64_t> > && sums)
            {
                std::int64_t sum = 0;
                for (auto & f : sums)
                    sum += f.get();
                return sum;
            },
            results);
    }
    return hpx::make_ready_future(num);
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    {
        std::uint64_t t = hpx::util::high_resolution_clock::now();

        hpx::future<std::int64_t> result = hpx::async(skynet, 0, 1000000, 10);
        result.wait();

        t = hpx::util::high_resolution_clock::now() - t;

        hpx::cout
            << "Result 1: " << result.get() << " in "
            << (t / 1e6) << " ms.\n";
    }

    {
        std::uint64_t t = hpx::util::high_resolution_clock::now();

        hpx::future<std::int64_t> result = hpx::async(skynet_f, 0, 1000000, 10);
        result.wait();

        t = hpx::util::high_resolution_clock::now() - t;

        hpx::cout
            << "Result 2: " << result.get() << " in "
            << (t / 1e6) << " ms.\n";
    }
    return 0;
}

