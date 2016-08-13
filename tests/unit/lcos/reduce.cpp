//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/lcos/reduce.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstdint>
#include <vector>

std::uint32_t f1()
{
    return hpx::get_locality_id();
}
HPX_PLAIN_ACTION(f1);

typedef std::plus<std::uint32_t> std_plus_type;
HPX_REGISTER_REDUCE_ACTION_DECLARATION(f1_action, std_plus_type)
HPX_REGISTER_REDUCE_ACTION(f1_action, std_plus_type)

std::uint32_t f3(std::uint32_t i)
{
    return hpx::get_locality_id() + i;
}
HPX_PLAIN_ACTION(f3);

HPX_REGISTER_REDUCE_ACTION_DECLARATION(f3_action, std_plus_type)
HPX_REGISTER_REDUCE_ACTION(f3_action, std_plus_type)

std::uint32_t f1_idx(std::size_t idx)
{
    return hpx::get_locality_id() + std::uint32_t(idx);
}
HPX_PLAIN_ACTION(f1_idx);

HPX_REGISTER_REDUCE_WITH_INDEX_ACTION_DECLARATION(f1_idx_action, std_plus_type)
HPX_REGISTER_REDUCE_WITH_INDEX_ACTION(f1_idx_action, std_plus_type)

std::uint32_t f3_idx(std::uint32_t i, std::size_t idx)
{
    return hpx::get_locality_id() + i + std::uint32_t(idx);
}
HPX_PLAIN_ACTION(f3_idx);

HPX_REGISTER_REDUCE_WITH_INDEX_ACTION_DECLARATION(f3_idx_action, std_plus_type)
HPX_REGISTER_REDUCE_WITH_INDEX_ACTION(f3_idx_action, std_plus_type)

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    hpx::id_type here = hpx::find_here();
    hpx::id_type there = here;
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    {
        std::uint32_t f1_res = hpx::lcos::reduce<f1_action>(
            localities, std::plus<std::uint32_t>()).get();

        std::uint32_t f1_result = 0;
        for(std::size_t i = 0; i != localities.size(); ++i)
        {
            f1_result += hpx::naming::get_locality_id_from_id(localities[i]);
        }
        HPX_TEST_EQ(f1_res, f1_result);

        std::uint32_t f3_res = hpx::lcos::reduce<f3_action>(
            localities, std::plus<std::uint32_t>(), 1).get();

        std::uint32_t f3_result = 0;
        for(std::size_t i = 0; i != localities.size(); ++i)
        {
            f3_result += hpx::naming::get_locality_id_from_id(localities[i]) + 1;
        }
        HPX_TEST_EQ(f3_res, f3_result);
    }

    {
        std::uint32_t f1_res = hpx::lcos::reduce_with_index<f1_idx_action>(
            localities, std::plus<std::uint32_t>()).get();

        std::uint32_t f1_result = 0;
        for(std::size_t i = 0; i != localities.size(); ++i)
        {
            f1_result += hpx::naming::get_locality_id_from_id(localities[i]) +
                std::uint32_t(i);
        }
        HPX_TEST_EQ(f1_res, f1_result);

        std::uint32_t f3_res = hpx::lcos::reduce_with_index<f3_idx_action>(
            localities, std::plus<std::uint32_t>(), 1).get();

        std::uint32_t f3_result = 0;
        for(std::size_t i = 0; i != localities.size(); ++i)
        {
            f3_result += hpx::naming::get_locality_id_from_id(localities[i]) +
                std::uint32_t(i) + 1;
        }
        HPX_TEST_EQ(f3_res, f3_result);
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

