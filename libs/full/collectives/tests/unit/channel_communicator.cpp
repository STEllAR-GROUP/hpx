//  Copyright (c) 2021 Hartmut Kaiser
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

#include <cstddef>
#include <utility>
#include <vector>

using namespace hpx::collectives;

///////////////////////////////////////////////////////////////////////////////
constexpr char const* channel_communicator_basename =
    "/test/channel_communicator/";
constexpr std::size_t NUM_CHANNEL_COMMUNICATOR_SITES = 32;

///////////////////////////////////////////////////////////////////////////////
void test_channel_communicator_set_first_comm(
    std::size_t site, hpx::collectives::channel_communicator comm)
{
    using data_type = std::pair<std::size_t, std::size_t>;

    // send a value to each of the participating sites
    std::vector<hpx::future<void>> sets;
    sets.reserve(NUM_CHANNEL_COMMUNICATOR_SITES);

    for (std::size_t i = 0; i != NUM_CHANNEL_COMMUNICATOR_SITES; ++i)
    {
        sets.push_back(set(comm, that_site_arg(i), std::make_pair(i, site)));
    }

    // receive the values sent above
    std::vector<hpx::future<data_type>> gets;
    gets.reserve(NUM_CHANNEL_COMMUNICATOR_SITES);

    for (std::size_t i = 0; i != NUM_CHANNEL_COMMUNICATOR_SITES; ++i)
    {
        gets.push_back(get<data_type>(comm, that_site_arg(i)));
    }

    hpx::wait_all(sets, gets);

    for (std::size_t i = 0; i != NUM_CHANNEL_COMMUNICATOR_SITES; ++i)
    {
        HPX_TEST(!sets[i].has_exception());

        auto data = gets[i].get();
        HPX_TEST_EQ(data.first, site);
        HPX_TEST_EQ(data.second, i);
    }
}

void test_channel_communicator_set_first_single_use(std::size_t site)
{
    // for each site, create new channel_communicator
    auto comm = create_channel_communicator(hpx::launch::sync,
        channel_communicator_basename,
        num_sites_arg(NUM_CHANNEL_COMMUNICATOR_SITES), this_site_arg(site));

    test_channel_communicator_set_first_comm(site, comm);
}

void test_single_use_set_first()
{
    for (std::size_t j = 0; j != 10; ++j)
    {
        std::vector<hpx::future<void>> tasks;
        tasks.reserve(NUM_CHANNEL_COMMUNICATOR_SITES);

        for (std::size_t i = 0; i != NUM_CHANNEL_COMMUNICATOR_SITES; ++i)
        {
            tasks.push_back(
                hpx::async(test_channel_communicator_set_first_single_use, i));
        }
        hpx::wait_all(tasks);
    }
}

void test_multi_use_set_first()
{
    for (std::size_t j = 0; j != 10; ++j)
    {
        // for each site, create new channel_communicator
        std::vector<hpx::collectives::channel_communicator> comms;
        comms.reserve(NUM_CHANNEL_COMMUNICATOR_SITES);

        for (std::size_t i = 0; i != NUM_CHANNEL_COMMUNICATOR_SITES; ++i)
        {
            comms.push_back(create_channel_communicator(hpx::launch::sync,
                channel_communicator_basename,
                num_sites_arg(NUM_CHANNEL_COMMUNICATOR_SITES),
                this_site_arg(i)));
        }

        std::vector<hpx::future<void>> tasks;
        tasks.reserve(NUM_CHANNEL_COMMUNICATOR_SITES);

        for (std::size_t i = 0; i != NUM_CHANNEL_COMMUNICATOR_SITES; ++i)
        {
            tasks.push_back(hpx::async(
                test_channel_communicator_set_first_comm, i, comms[i]));
        }

        hpx::wait_all(tasks);
    }
}

///////////////////////////////////////////////////////////////////////////////
void test_channel_communicator_get_first_comm(
    std::size_t site, hpx::collectives::channel_communicator comm)
{
    using data_type = std::pair<std::size_t, std::size_t>;

    // receive the values sent above
    std::vector<hpx::future<data_type>> gets;
    gets.reserve(NUM_CHANNEL_COMMUNICATOR_SITES);

    for (std::size_t i = 0; i != NUM_CHANNEL_COMMUNICATOR_SITES; ++i)
    {
        gets.push_back(get<data_type>(comm, that_site_arg(i)));
    }

    // send a value to each of the participating sites
    std::vector<hpx::future<void>> sets;
    sets.reserve(NUM_CHANNEL_COMMUNICATOR_SITES);

    for (std::size_t i = 0; i != NUM_CHANNEL_COMMUNICATOR_SITES; ++i)
    {
        sets.push_back(set(comm, that_site_arg(i), std::make_pair(i, site)));
    }

    hpx::wait_all(sets, gets);

    for (std::size_t i = 0; i != NUM_CHANNEL_COMMUNICATOR_SITES; ++i)
    {
        HPX_TEST(!sets[i].has_exception());

        auto data = gets[i].get();
        HPX_TEST_EQ(data.first, site);
        HPX_TEST_EQ(data.second, i);
    }
}

void test_channel_communicator_get_first_single_use(std::size_t site)
{
    // for each site, create new channel_communicator
    auto comm = create_channel_communicator(hpx::launch::sync,
        channel_communicator_basename,
        num_sites_arg(NUM_CHANNEL_COMMUNICATOR_SITES), this_site_arg(site));

    test_channel_communicator_get_first_comm(site, comm);
}

void test_single_use_get_first()
{
    for (std::size_t j = 0; j != 10; ++j)
    {
        std::vector<hpx::future<void>> tasks;
        tasks.reserve(NUM_CHANNEL_COMMUNICATOR_SITES);

        for (std::size_t i = 0; i != NUM_CHANNEL_COMMUNICATOR_SITES; ++i)
        {
            tasks.push_back(
                hpx::async(test_channel_communicator_get_first_single_use, i));
        }

        hpx::wait_all(tasks);
    }
}

void test_multi_use_get_first()
{
    for (std::size_t j = 0; j != 10; ++j)
    {
        // for each site, create new channel_communicator
        std::vector<hpx::collectives::channel_communicator> comms;
        comms.reserve(NUM_CHANNEL_COMMUNICATOR_SITES);

        for (std::size_t i = 0; i != NUM_CHANNEL_COMMUNICATOR_SITES; ++i)
        {
            comms.push_back(create_channel_communicator(hpx::launch::sync,
                channel_communicator_basename,
                num_sites_arg(NUM_CHANNEL_COMMUNICATOR_SITES),
                this_site_arg(i)));
        }

        std::vector<hpx::future<void>> tasks;
        tasks.reserve(NUM_CHANNEL_COMMUNICATOR_SITES);

        for (std::size_t i = 0; i != NUM_CHANNEL_COMMUNICATOR_SITES; ++i)
        {
            tasks.push_back(hpx::async(
                test_channel_communicator_get_first_comm, i, comms[i]));
        }

        hpx::wait_all(tasks);
    }
}

int hpx_main()
{
    test_single_use_set_first();
    test_single_use_get_first();

    test_multi_use_set_first();
    test_multi_use_get_first();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
#endif
