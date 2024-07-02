//  Copyright (c) 2019-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/testing.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace hpx::collectives;

constexpr char const* all_reduce_direct_basename = "/test/all_reduce_direct/";
#if defined(HPX_DEBUG)
constexpr int ITERATIONS = 100;
#else
constexpr int ITERATIONS = 1000;
#endif

struct plus_bool
{
    template <typename T>
    decltype(auto) operator()(T lhs, T rhs) const
    {
        return lhs + rhs;
    }
};

void test_multiple_use_with_generation()
{
    std::uint32_t const this_locality = hpx::get_locality_id();
    std::uint32_t const num_localities =
        hpx::get_num_localities(hpx::launch::sync);
    HPX_TEST_LTE(static_cast<std::uint32_t>(2), num_localities);

    auto const all_reduce_direct_client =
        create_communicator(all_reduce_direct_basename,
            num_sites_arg(num_localities), this_site_arg(this_locality));

    for (int i = 0; i != ITERATIONS; ++i)
    {
        bool value = ((this_locality + i) % 2) ? true : false;
        hpx::future<bool> overall_result = all_reduce(all_reduce_direct_client,
            value, plus_bool{}, generation_arg(i + 1));

        bool sum = false;
        for (std::uint32_t j = 0; j != num_localities; ++j)
        {
            sum = sum + (((j + i) % 2) ? true : false);
        }
        HPX_TEST_EQ(sum, overall_result.get());
    }
}

void test_local_use()
{
    constexpr std::uint32_t num_sites = 10;

    std::vector<hpx::future<void>> sites;
    sites.reserve(num_sites);

    // launch num_sites threads to represent different sites
    for (std::uint32_t site = 0; site != num_sites; ++site)
    {
        sites.push_back(hpx::async([=]() {
            auto const all_reduce_direct_client =
                create_communicator(all_reduce_direct_basename,
                    num_sites_arg(num_sites), this_site_arg(site));

            // test functionality based on immediate local result value
            for (int i = 0; i != ITERATIONS; ++i)
            {
                bool value = ((site + i) % 2) ? true : false;
                hpx::future<bool> result =
                    all_reduce(all_reduce_direct_client, value, std::plus<>{},
                        this_site_arg(site), generation_arg(i + 1));

                bool sum = false;
                for (std::uint32_t j = 0; j != num_sites; ++j)
                {
                    sum = sum + (((j + i) % 2) ? true : false);
                }
                HPX_TEST_EQ(sum, result.get());
            }
        }));
    }

    hpx::wait_all(std::move(sites));
}

int hpx_main()
{
#if defined(HPX_HAVE_NETWORKING)
    if (hpx::get_num_localities(hpx::launch::sync) > 1)
    {
        test_multiple_use_with_generation();
    }
#endif

    if (hpx::get_locality_id() == 0)
    {
        test_local_use();
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {"hpx.run_hpx_main!=1"};

    hpx::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);
    return hpx::util::report_errors();
}

#endif
