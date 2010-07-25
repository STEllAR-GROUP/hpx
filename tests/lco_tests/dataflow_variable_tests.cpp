//  Copyright (c) 2010-2011 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

using namespace hpx;
namespace po = boost::program_options;

///////////////////////////////////////////////////////////////////////////////
// Helpers

typedef hpx::naming::id_type id_type;
typedef hpx::naming::gid_type gid_type;

inline gid_type find_here(void)
{
    return hpx::applier::get_applier().get_runtime_support_raw_gid();
}

inline gid_type find_there(void)
{
    std::vector<gid_type> localities;
    applier::get_applier().get_remote_prefixes(localities);

    if (localities.size() > 0)
        return localities[0];
    else
        return find_here();
}

///////////////////////////////////////////////////////////////////////////////
int print(id_type d_id)
{
  typedef lcos::base_lco_with_value<int>::get_value_action get_action;

  std::cout << "print> print d" << std::endl;
  std::cout << lcos::eager_future<get_action>(d_id).get() << std::endl;

  return 0;
}
typedef actions::plain_result_action1<int, id_type, print> print_action;
HPX_REGISTER_PLAIN_ACTION(print_action);

///////////////////////////////////////////////////////////////////////////////
typedef hpx::lcos::dataflow_variable<int,int> dataflow_int_type;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(po::variables_map &vm)
{
    id_type here = naming::id_type(find_here(), naming::id_type::unmanaged);
    id_type there = naming::id_type(find_there(), naming::id_type::unmanaged);

    std::cout << ">>> print here, there" << std::endl;
    std::cout << here << " " << there << std::endl << std::endl;

    {
        std::cout << ">>> d1 = dataflow_variable()" << std::endl;
        dataflow_int_type d1;

        std::cout << ">>> spawning { print d1 } here" << std::endl; 
        lcos::future_value<int> 
            local_spawn(lcos::eager_future<print_action>(here, d1.get_gid()));

        std::cout << ">>> spawning { print d1 } there" << std::endl; 
        lcos::future_value<int> 
            remote_spawn(lcos::eager_future<print_action>(there, d1.get_gid()));

        std::cout << ">>> bind(d1, 42)" << std::endl;
        actions::continuation(d1.get_gid()).trigger<int>(42);

        // We have to use the eager_future here in order to make sure the
        // spawned action has completed before we let the dataflow_variable
        // go out of scope and destruct.
        local_spawn.get();
        remote_spawn.get();
    }

    // initiate shutdown of the runtime systems on all localities
    components::stubs::runtime_support::shutdown_all();

    return 0;
}


///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{

  // Configure application-specific options
  po::options_description
      desc_commandline(
          "Usage: dataflow_variable_tests [hpx_options] [options]");

  // Initialize and run HPX
  int retcode = hpx_init(desc_commandline, argc, argv);
  return retcode;
}

