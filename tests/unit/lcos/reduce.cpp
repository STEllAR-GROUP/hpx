//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/lcos/reduce.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <vector>

boost::uint32_t f1()
{
    return hpx::get_locality_id();
}
HPX_PLAIN_ACTION(f1);

typedef std::plus<boost::uint32_t> std_plus_type;
HPX_REGISTER_REDUCE_ACTION_DECLARATION(f1_action, std_plus_type)
HPX_REGISTER_REDUCE_ACTION(f1_action, std_plus_type)

boost::uint32_t f3(boost::uint32_t i)
{
    return hpx::get_locality_id() + i;
}
HPX_PLAIN_ACTION(f3);

HPX_REGISTER_REDUCE_ACTION_DECLARATION(f3_action, std_plus_type)
HPX_REGISTER_REDUCE_ACTION(f3_action, std_plus_type)

boost::uint32_t f1_idx(std::size_t idx)
{
    return hpx::get_locality_id() + boost::uint32_t(idx);
}
HPX_PLAIN_ACTION(f1_idx);

HPX_REGISTER_REDUCE_WITH_INDEX_ACTION_DECLARATION(f1_idx_action, std_plus_type)
HPX_REGISTER_REDUCE_WITH_INDEX_ACTION(f1_idx_action, std_plus_type)

boost::uint32_t f3_idx(boost::uint32_t i, std::size_t idx)
{
    return hpx::get_locality_id() + i + boost::uint32_t(idx);
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
        boost::uint32_t f1_res = hpx::lcos::reduce<f1_action>(
            localities, std::plus<boost::uint32_t>()).get();

        boost::uint32_t f1_result = 0;
        for(std::size_t i = 0; i != localities.size(); ++i)
        {
            f1_result += hpx::naming::get_locality_id_from_id(localities[i]);
        }
        HPX_TEST_EQ(f1_res, f1_result);

        boost::uint32_t f3_res = hpx::lcos::reduce<f3_action>(
            localities, std::plus<boost::uint32_t>(), 1).get();

        boost::uint32_t f3_result = 0;
        for(std::size_t i = 0; i != localities.size(); ++i)
        {
            f3_result += hpx::naming::get_locality_id_from_id(localities[i]) + 1;
        }
        HPX_TEST_EQ(f3_res, f3_result);
    }

    {
        boost::uint32_t f1_res = hpx::lcos::reduce_with_index<f1_idx_action>(
            localities, std::plus<boost::uint32_t>()).get();

        boost::uint32_t f1_result = 0;
        for(std::size_t i = 0; i != localities.size(); ++i)
        {
            f1_result += hpx::naming::get_locality_id_from_id(localities[i]) +
                boost::uint32_t(i);
        }
        HPX_TEST_EQ(f1_res, f1_result);

        boost::uint32_t f3_res = hpx::lcos::reduce_with_index<f3_idx_action>(
            localities, std::plus<boost::uint32_t>(), 1).get();

        boost::uint32_t f3_result = 0;
        for(std::size_t i = 0; i != localities.size(); ++i)
        {
            f3_result += hpx::naming::get_locality_id_from_id(localities[i]) +
                boost::uint32_t(i) + 1;
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

