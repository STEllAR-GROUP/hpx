//  Copyright (c) 2010-2011 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <ctime>
#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>

#include <hpx/util/random.hpp>

#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

using namespace hpx;
namespace po = boost::program_options;

////////////////////////////////////////////////////////////////////////////////
typedef int size_type;
typedef lcos::future_value<size_type> future_size_type;
typedef std::vector<future_size_type> future_sizes_type;

////////////////////////////////////////////////////////////////////////////////
size_type hits(size_type throws)
{
  using namespace util::random;

  random_generator random_value;
  random_value.seed((unsigned int)time(0));

  size_type num_hits = 0;

  for (size_type i=0; i<throws; i++)
  {
    double x = (double)random_value() / get_max();
    double y = (double)random_value() / get_max();

    if (x*x + y*y <= 1.0)
      num_hits++;
  }

  std::string out(
      boost::lexical_cast<std::string>(num_hits)
      + " hits for "
      + boost::lexical_cast<std::string>(throws)
      + " throws\n");
  std::cout << out;

  return num_hits;
}
typedef actions::plain_result_action1<size_type, size_type, hits> hits_action;
HPX_REGISTER_PLAIN_ACTION(hits_action);

size_type local_hits(size_type throws, size_type granularity)
{
  size_type const local_throws = throws / granularity;

  gid_type here = find_here();

  future_sizes_type hits;
  for (size_type i=0; i < granularity; i++)
  {
    hits.push_back(lcos::eager_future<hits_action>(here, local_throws));
  }

  size_type total_hits = 0;
  while (hits.size() > 0)
  {
    total_hits += hits.back().get();
    hits.pop_back();
  }

  return total_hits;
}
typedef actions::plain_result_action2<
  size_type, size_type, size_type, local_hits
> local_hits_action;
HPX_REGISTER_PLAIN_ACTION(local_hits_action);

////////////////////////////////////////////////////////////////////////////////
int hpx_main(po::variables_map &vm)
{
  size_type throws = 1000;
  size_type granularity = 1;

  get_option(vm, "throws", throws);
  get_option(vm, "granularity", granularity);

  process my_proc(get_runtime().get_process());

  {
    util::high_resolution_timer t;

    size_type const local_throws = throws / my_proc.size();

    future_sizes_type hits;
    for (size_type lid = 0; lid < my_proc.size(); lid++)
    {
      hits.push_back(
          lcos::eager_future<local_hits_action>(
            my_proc.there(lid), local_throws, granularity));
    }

    size_type total_hits = 0;
    while (hits.size() > 0)
    {
      total_hits += hits.back().get();
      hits.pop_back();
    }

    size_type total_misses = throws-total_hits;

    double pi = 4.0*total_hits/throws;

    std::cout << "throws = " << throws << std::endl
      << "total_hits = " << total_hits << std::endl
      << "pi = " <<  pi << std::endl;

    std::cout << "Elapsed time: " << t.elapsed() << std::endl;
  }

  hpx_finalize();

  std::cout << "Test passed" << std::endl;

  return 0;
}

int main(int argc, char* argv[])
{
  po::options_description
    desc_commandline("Usage: cpi_test [hpx_options] [options]");
  desc_commandline.add_options()
    ("throws", po::value<size_type>(),
     "The number of throws to attempt (default: 1000)")
    ("granularity", po::value<size_type>(),
     "The number of concurrent throwers to use per locality (default: 1)")
    ;

  int retcode = hpx_init(desc_commandline, argc, argv);
  return retcode;
}
