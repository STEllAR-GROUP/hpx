//  Copyright (c) 2024 Hartmut Kaiser
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

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

using Dimension = unsigned short;

template <typename TYPE, Dimension DIM, std::size_t NAMESPACE>
class dimensioned_array
{
public:
    dimensioned_array() = default;

    explicit constexpr dimensioned_array(TYPE const& val)
      : data_()
    {
        for (auto& x : data_)
            x = val;
    }

    constexpr bool operator==(dimensioned_array const& da)
    {
        return this->data_ == da.data_;
    }

private:
    std::array<TYPE, DIM> data_;
};

template <typename TYPE, Dimension DIM>
using point = dimensioned_array<TYPE, DIM, 1>;

template <Dimension DIM>
struct BBox
{
    point<double, DIM> lower, upper;
};

void test_local_use()
{
    static_assert(hpx::traits::is_bitwise_serializable_v<point<double, 2>>,
        "hpx::traits::is_bitwise_serializable_v<point<double, 2>>");
    static_assert(hpx::traits::is_bitwise_serializable_v<BBox<2>>,
        "hpx::traits::is_bitwise_serializable_v<point<double, 2>>");

    using namespace hpx::collectives;

    constexpr char const* all_gather_direct_basename =
        "/test/all_gather_direct/";
    constexpr int ITERATIONS = 100;
    constexpr std::uint32_t num_sites = 10;

    std::vector<hpx::future<void>> sites;
    sites.reserve(num_sites);

    // launch num_sites threads to represent different sites
    for (std::uint32_t site = 0; site != num_sites; ++site)
    {
        sites.push_back(hpx::async([=]() {
            auto const all_gather_direct_client =
                create_local_communicator(all_gather_direct_basename,
                    num_sites_arg(num_sites), this_site_arg(site));

            for (std::uint32_t i = 0; i != 10 * ITERATIONS; ++i)
            {
                double const coord_x = i;
                double const coord_y = i + 1;
                hpx::future<std::vector<BBox<2>>> overall_result =
                    all_gather(all_gather_direct_client,
                        BBox<2>{point<double, 2>{coord_x},
                            point<double, 2>{coord_y}},
                        this_site_arg(site), generation_arg(i + 1));

                std::vector<BBox<2>> r = overall_result.get();
                HPX_TEST_EQ(r.size(), num_sites);
            }
        }));
    }

    hpx::wait_all(std::move(sites));
}

int hpx_main()
{
    test_local_use();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}

#endif
