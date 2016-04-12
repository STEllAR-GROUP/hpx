//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The purpose of this example is to demonstrate how to disable HPX command
// line alias handling and to allow for unknown options to be passed through
// to hpx_main.

#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>

#include <string>
#include <vector>

int hpx_main(int argc, char* argv[])
{
    for (int i = 0; i != argc; ++i)
        hpx::cout << "arg(" << i << "): " << argv[i] << hpx::endl;

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg;
    cfg.push_back("hpx.commandline.aliasing=0");        // disable aliasing
    cfg.push_back("hpx.commandline.allow_unknown=1");   // allow for unknown options

    return hpx::init(argc, argv, cfg);
}

