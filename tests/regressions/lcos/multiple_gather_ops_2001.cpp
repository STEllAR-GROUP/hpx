//  Copyright (c) 2016 Lukas Troska
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #2001: Gathering more
// than once segfaults

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>

#include <boost/cstdint.hpp>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

char const* gather_basename = "/test/gather/";

HPX_REGISTER_GATHER(boost::uint32_t, test_gather);

int hpx_main(int argc, char* argv[])
{
    for (int i = 0; i < 10; ++i)
    {
        hpx::future<boost::uint32_t> value =
            hpx::make_ready_future(hpx::get_locality_id());

        if (hpx::get_locality_id() == 0)
        {
            hpx::future<std::vector<boost::uint32_t> > overall_result =
                hpx::lcos::gather_here(gather_basename, std::move(value),
                    hpx::get_num_localities(hpx::launch::sync), i);

            std::vector<boost::uint32_t> sol = overall_result.get();

            for (std::size_t j = 0; j < sol.size(); ++j)
            {
                std::cout << "got residual " << sol[j]  << " from " << j
                    << std::endl;
            }
        }
        else
        {
            hpx::future<void> f = hpx::lcos::gather_there(gather_basename,
                std::move(value), i);
            f.get();
        }
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> const cfg = {
        "hpx.run_hpx_main!=1"
    };
    return hpx::init(argc, argv, cfg);
}
