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
    std::cout << hpx::get_locality_id() << "\n";
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
            std::cout << i << " " << f1_res[i] << " " << hpx::naming::get_locality_id_from_id(localities[i]) << "\n";
            //HPX_TEST_EQ(f1_res[i], hpx::naming::get_locality_id_from_id(localities[i]));
        }
        
        hpx::lcos::broadcast<f2_action>(localities).get();
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

