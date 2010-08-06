//  Copyright (c) 2010-2011 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <boost/program_options.hpp>

using namespace hpx;
namespace po = boost::program_options;

////////////////////////////////////////////////////////////////////////////////
int hpx_main(po::variables_map &vm)
{
  hpx_finalize();

  std::cout << "Test passed" << std::endl;

  return 0;
}

int main(int argc, char* argv[])
{
  po::options_description
    desc_commandline("Usage: start_up_test [hpx_options]");

  int retcode = hpx_init(desc_commandline, argc, argv);
  return retcode;
}
