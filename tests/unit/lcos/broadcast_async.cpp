//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/lcos/broadcast_async.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <vector>

HPX_CONSTEXPR char const* const broadcast_basename = "/broadcast_async_test";
HPX_CONSTEXPR int const num_sites = 10;
HPX_CONSTEXPR int const num_generations = 100;

using std_vector_int = std::vector<int>;
HPX_BROADCAST_ASYNC(std_vector_int);

int hpx_main()
{
    std::vector<int> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

    for (int i = 0; i != num_generations; ++i)
    {
        hpx::future<void> there =
            hpx::lcos::broadcast_there(broadcast_basename, data, num_sites, i);

        std::vector<hpx::future<void> > futures;
        futures.reserve(num_sites);
        for (int j = 0; j != num_sites; ++j)
        {
            futures.push_back(hpx::async(
                [&, j]() -> void
                {
                    std::vector<int> result =
                        hpx::lcos::broadcast_here<std::vector<int> >(
                            broadcast_basename, j, i
                        ).get();

                    HPX_TEST(
                        std::equal(result.begin(), result.end(), data.begin())
                    );
                }));
        }
        hpx::wait_all(futures);

        there.get();
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}

