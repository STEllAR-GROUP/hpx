////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>

#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>
#include <hpx/iostream.hpp>
#ifdef HPX_HAVE_COMPRESSION_ZLIB
#include <hpx/include/compression_zlib.hpp>
#endif
#include <hpx/include/parcel_coalescing.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <vector>

using hpx::program_options::options_description;
using hpx::program_options::variables_map;

struct functor
{
    functor() {}

    int operator()() const
    {
        return 42;
    }
};

int pass_functor(hpx::util::function<int()> const& f)
{
    return f();
}

HPX_DECLARE_PLAIN_ACTION(pass_functor, pass_functor_action);
#ifdef HPX_HAVE_COMPRESSION_ZLIB
HPX_ACTION_USES_ZLIB_COMPRESSION(pass_functor_action);
#endif
HPX_ACTION_USES_MESSAGE_COALESCING(pass_functor_action);
HPX_PLAIN_ACTION(pass_functor, pass_functor_action);

void worker(hpx::util::function<int()> const& f)
{
    pass_functor_action act;

    std::vector<hpx::id_type> targets = hpx::find_remote_localities();

    for (std::size_t j = 0; j != 100; ++j)
    {
        for (std::size_t i = 0; i < targets.size(); ++i)
        {
            HPX_TEST_EQ(act(targets[i], f), 42);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    hpx::chrono::high_resolution_timer t;

    {
        functor g;
        hpx::util::function<int()> f(g);

        std::vector<hpx::future<void>> futures;

        for (std::size_t i = 0; i != 16; ++i)
        {
            futures.push_back(hpx::async(&worker, f));
        }

        hpx::wait_all(futures);
    }

    double elapsed = t.elapsed();
    hpx::cout << "Elapsed time: " << elapsed << "\n" << hpx::flush;

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);
    return 0;
}
#endif
