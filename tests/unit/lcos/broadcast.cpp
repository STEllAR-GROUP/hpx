//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/lcos/broadcast.hpp>
#include <hpx/util/lightweight_test.hpp>

boost::uint32_t f1()
{
    return hpx::get_locality_id();
}
HPX_PLAIN_ACTION(f1);

HPX_REGISTER_BROADCAST_ACTION_DECLARATION(f1_action)
HPX_REGISTER_BROADCAST_ACTION(f1_action)

void f2()
{
}
HPX_PLAIN_ACTION(f2);

HPX_REGISTER_BROADCAST_ACTION_DECLARATION(f2_action)
HPX_REGISTER_BROADCAST_ACTION(f2_action)

boost::uint32_t f3(boost::uint32_t i)
{
    return hpx::get_locality_id() + i;
}
HPX_PLAIN_ACTION(f3);

HPX_REGISTER_BROADCAST_ACTION_DECLARATION(f3_action)
HPX_REGISTER_BROADCAST_ACTION(f3_action)

void f4(boost::uint32_t i)
{
}
HPX_PLAIN_ACTION(f4);

HPX_REGISTER_BROADCAST_ACTION_DECLARATION(f4_action)
HPX_REGISTER_BROADCAST_ACTION(f4_action)

int hpx_main()
{
    hpx::id_type here = hpx::find_here();
    hpx::id_type there = here;
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    {
        std::vector<boost::uint32_t> f1_res;
        f1_res = hpx::lcos::broadcast<f1_action>(localities).get();

        HPX_TEST_EQ(f1_res.size(), localities.size());
        for(std::size_t i = 0; i < f1_res.size(); ++i)
        {
            HPX_TEST_EQ(f1_res[i], hpx::naming::get_locality_id_from_id(localities[i]));
        }
        
        hpx::lcos::broadcast<f2_action>(localities).get();
        
        std::vector<boost::uint32_t> f3_res;
        f3_res = hpx::lcos::broadcast<f3_action>(localities, 1).get();

        HPX_TEST_EQ(f3_res.size(), localities.size());
        for(std::size_t i = 0; i < f3_res.size(); ++i)
        {
            HPX_TEST_EQ(f3_res[i], hpx::naming::get_locality_id_from_id(localities[i]) + 1);
        }
        
        hpx::lcos::broadcast<f4_action>(localities, 0).get();
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

