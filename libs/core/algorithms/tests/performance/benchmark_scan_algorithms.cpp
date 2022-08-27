//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/init.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/algorithms/exclusive_scan.hpp>
#include <hpx/parallel/algorithms/inclusive_scan.hpp>
#include <hpx/parallel/algorithms/transform_exclusive_scan.hpp>
#include <hpx/parallel/algorithms/transform_inclusive_scan.hpp>
#include <hpx/parallel/algorithms/unique.hpp>

#include <array>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

//#define OUTPUT_TO_CSV

enum class ALGORITHM
{
    EXCLUSIVE_SCAN,
    INCLUSIVE_SCAN,
    TRANSFORM_EXCLUSIVE_SCAN,
    TRANSFORM_INCLUSIVE_SCAN,
    COPY_IF,
    UNIQUE_COPY
};

void measureScanAlgorithms()
{
#if defined(OUTPUT_TO_CSV)
    std::map<ALGORITHM, std::string> filenames = {
        {ALGORITHM::EXCLUSIVE_SCAN, "exclusiveScanCompare.csv"},
        {ALGORITHM::INCLUSIVE_SCAN, "inclusiveScanCompare.csv"},
        {ALGORITHM::TRANSFORM_EXCLUSIVE_SCAN,
            "transformExclusiveScanCompare.csv"},
        {ALGORITHM::TRANSFORM_INCLUSIVE_SCAN,
            "transformInclusiveScanCompare.csv"},
        {ALGORITHM::COPY_IF, "copyIfCompare.csv"},
        {ALGORITHM::UNIQUE_COPY, "uniqueCopyCompare.csv"},
    };
#endif

    for (int alg = (int) ALGORITHM::EXCLUSIVE_SCAN;
         alg <= (int) ALGORITHM::UNIQUE_COPY; alg++)
    {
        std::size_t start = 32;
        std::size_t till = 1 << 10;

        const auto NUM_ITERATIONS = 5;

        std::vector<std::array<double, 3>> data;

        for (std::size_t s = start; s <= till; s *= 2)
        {
            std::vector<int> arr(s);
            std::iota(std::begin(arr), std::end(arr), 1);

            double seqTime = 0;
            double parTime = 0;

            for (int i = 0; i < NUM_ITERATIONS + 5; i++)
            {
                std::vector<int> res(s);
                auto t1 = std::chrono::high_resolution_clock::now();

                switch ((ALGORITHM) alg)
                {
                case ALGORITHM::INCLUSIVE_SCAN:
                    hpx::inclusive_scan(arr.begin(), arr.end(), res.begin(),
                        std::plus<int>(), 0);
                    break;
                case ALGORITHM::EXCLUSIVE_SCAN:
                    hpx::exclusive_scan(arr.begin(), arr.end(), res.begin(), 10,
                        std::plus<int>{});
                    break;
                case ALGORITHM::TRANSFORM_EXCLUSIVE_SCAN:
                    hpx::transform_exclusive_scan(arr.begin(), arr.end(),
                        res.begin(), 10, std::plus<int>{},
                        [](int x) { return x * 10; });
                    break;
                case ALGORITHM::TRANSFORM_INCLUSIVE_SCAN:
                    hpx::transform_inclusive_scan(
                        arr.begin(), arr.end(), res.begin(), std::plus<int>{},
                        [](int x) { return x * 10; }, 10);
                    break;
                case ALGORITHM::COPY_IF:
                    hpx::copy_if(arr.begin(), arr.end(), res.begin(),
                        [](int x) { return (x % 2) != 0; });
                    break;
                case ALGORITHM::UNIQUE_COPY:
                    hpx::unique_copy(arr.begin(), arr.end(), res.begin(),
                        std::equal_to<int>{});
                    break;
                };
                auto end1 = std::chrono::high_resolution_clock::now();

                // don't consider first 5 iterations
                if (NUM_ITERATIONS < 5)
                {
                    continue;
                }

                std::chrono::duration<double> time_span1 =
                    std::chrono::duration_cast<std::chrono::duration<double>>(
                        end1 - t1);

                seqTime += time_span1.count();
            }

            for (int i = 0; i < NUM_ITERATIONS + 5; i++)
            {
                std::vector<int> res1(s);
                auto t2 = std::chrono::high_resolution_clock::now();
                switch ((ALGORITHM) alg)
                {
                case ALGORITHM::INCLUSIVE_SCAN:
                    hpx::inclusive_scan(hpx::execution::par, arr.begin(),
                        arr.end(), res1.begin(), std::plus<int>(), 0);
                    break;
                case ALGORITHM::EXCLUSIVE_SCAN:
                    hpx::exclusive_scan(hpx::execution::par, arr.begin(),
                        arr.end(), res1.begin(), 10, std::plus<int>{});
                    break;
                case ALGORITHM::TRANSFORM_EXCLUSIVE_SCAN:
                    hpx::transform_exclusive_scan(hpx::execution::par,
                        arr.begin(), arr.end(), res1.begin(), 10,
                        std::plus<int>{}, [](int x) { return x * 10; });
                    break;
                case ALGORITHM::TRANSFORM_INCLUSIVE_SCAN:
                    hpx::transform_inclusive_scan(
                        hpx::execution::par, arr.begin(), arr.end(),
                        res1.begin(), std::plus<int>{},
                        [](int x) { return x * 10; }, 10);
                    break;
                case ALGORITHM::COPY_IF:
                    hpx::copy_if(hpx::execution::par, arr.begin(), arr.end(),
                        res1.begin(), [](int x) { return (x % 2) != 0; });
                    break;
                case ALGORITHM::UNIQUE_COPY:
                    hpx::unique_copy(hpx::execution::par, arr.begin(),
                        arr.end(), res1.begin(), std::equal_to<int>{});
                    break;
                };
                auto end2 = std::chrono::high_resolution_clock::now();

                // don't consider first 5 iterations
                if (NUM_ITERATIONS < 5)
                {
                    continue;
                }

                std::chrono::duration<double> time_span2 =
                    std::chrono::duration_cast<std::chrono::duration<double>>(
                        end2 - t2);

                parTime += time_span2.count();
            }

            seqTime /= NUM_ITERATIONS;
            parTime /= NUM_ITERATIONS;

#if defined(OUTPUT_TO_CSV)
            data.push_back(std::array<double, 3>{(double) s, seqTime, parTime});
#else
            std::cout << "N : " << s << '\n';
            std::cout << "SEQ: " << seqTime << '\n';
            std::cout << "PAR: " << parTime << "\n\n";
#endif
        }

#if defined(OUTPUT_TO_CSV)
        std::ofstream outputFile(filenames[(ALGORITHM) alg]);
        for (auto& d : data)
        {
            outputFile << d[0] << "," << d[1] << "," << d[2] << ","
                       << ",\n";
        }
#endif
    }
}

int hpx_main(hpx::program_options::variables_map&)
{
    measureScanAlgorithms();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=all");
    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    // Initialize and run HPX.
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
