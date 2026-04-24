//  Copyright (c) 2026 Anshuman Agrawal
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
#include <functional>
#include <string>
#include <utility>
#include <vector>

using namespace hpx::collectives;

// Distributed test (multi-locality): verifies that the fallback path produces
// size() == 1 and correct collective results, and that threshold == 0 forces
// the tree path with the same results.
void test_distributed_fallback()
{
    std::uint32_t const this_locality = hpx::get_locality_id();
    std::uint32_t const num_localities =
        hpx::get_num_localities(hpx::launch::sync);
    HPX_TEST_LTE(static_cast<std::uint32_t>(2), num_localities);

    // Force the fallback path by setting threshold above num_localities.
    auto const fb_clients = create_hierarchical_communicator(
        "/test/hflback_dist_fb/", num_sites_arg(num_localities),
        this_site_arg(this_locality), arity_arg(2), generation_arg(),
        root_site_arg(0), flat_fallback_threshold_arg(1024));

    HPX_TEST_EQ(fb_clients.size(), static_cast<std::size_t>(1));

    std::uint32_t const value = this_locality + 1;
    std::uint32_t const fb_result =
        all_reduce(fb_clients, std::uint32_t(value), std::plus<std::uint32_t>{},
            this_site_arg(this_locality), generation_arg(1))
            .get();

    std::uint32_t expected = 0;
    for (std::uint32_t j = 0; j != num_localities; ++j)
    {
        expected += (j + 1);
    }
    HPX_TEST_EQ(fb_result, expected);

    // Force the tree path by setting threshold to 0; results must match.
    auto const tree_clients = create_hierarchical_communicator(
        "/test/hflback_dist_tree/", num_sites_arg(num_localities),
        this_site_arg(this_locality), arity_arg(2), generation_arg(),
        root_site_arg(0), flat_fallback_threshold_arg(0));

    std::uint32_t const tree_result = all_reduce(tree_clients,
        std::uint32_t(value), std::plus<std::uint32_t>{},
        this_site_arg(this_locality), generation_arg(1))
                                          .get();

    HPX_TEST_EQ(tree_result, expected);
}

// Local test (single-process, multi-thread sites): verifies that the fallback
// works for a range of site counts below the default threshold, exercised
// from locality 0 only.
void test_local_fallback(std::uint32_t num_sites)
{
    std::vector<hpx::future<void>> sites;
    sites.reserve(num_sites);

    for (std::uint32_t site = 0; site != num_sites; ++site)
    {
        sites.push_back(hpx::async([=]() {
            auto const clients = create_hierarchical_communicator(
                "/test/hflback_local/", num_sites_arg(num_sites),
                this_site_arg(site), arity_arg(2), generation_arg(),
                root_site_arg(0), flat_fallback_threshold_arg(1024));

            HPX_TEST_EQ(clients.size(), static_cast<std::size_t>(1));

            std::uint32_t const value = site + 1;
            std::uint32_t const result = all_reduce(clients,
                std::uint32_t(value), std::plus<std::uint32_t>{},
                this_site_arg(site), generation_arg(1))
                                             .get();

            std::uint32_t expected = 0;
            for (std::uint32_t j = 0; j != num_sites; ++j)
            {
                expected += (j + 1);
            }
            HPX_TEST_EQ(result, expected);
        }));
    }

    hpx::wait_all(std::move(sites));
}

int hpx_main()
{
#if defined(HPX_HAVE_NETWORKING)
    if (hpx::get_num_localities(hpx::launch::sync) > 1)
    {
        test_distributed_fallback();
    }
#endif

    if (hpx::get_locality_id() == 0)
    {
        for (std::uint32_t n : {2u, 4u, 8u})
        {
            test_local_fallback(n);
        }
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
