////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// Naive SMP version implemented with futures.

#include <hpx/hpx.hpp>
#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/lcos/async.hpp>

#include <iostream>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::naming::id_type;

using hpx::applier::get_applier;

using hpx::actions::plain_result_action2;

using hpx::lcos::async;

using hpx::util::high_resolution_timer;

using hpx::init;
using hpx::finalize;

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t factorial(
    id_type const& prefix
  , boost::uint64_t m
);

typedef plain_result_action2<
    // result type
    boost::uint64_t
    // arguments
  , id_type const&
  , boost::uint64_t
    // function
  , factorial
> factorial_action;

HPX_REGISTER_PLAIN_ACTION(factorial_action);

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t factorial(
    id_type const& prefix
  , boost::uint64_t n
) {
    if (0 >= n)
        return 1;

    else
    {
        future<boost::uint64_t> n1 =
            async<factorial_action>(prefix, prefix, n - 1);
        return n * n1.get();
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        boost::uint64_t n = vm["n-value"].as<boost::uint64_t>();

        const id_type prefix = get_applier().get_runtime_support_gid();

        high_resolution_timer t;

        boost::uint64_t r = async<factorial_action>(prefix, prefix, n).get();

        double elapsed = t.elapsed();

        std::cout
            << ( boost::format("factorial(%1%) == %2%\n"
                               "elapsed time == %3% [s]\n")
               % n % r % elapsed);
    }

    finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "n-value"
        , value<boost::uint64_t>()->default_value(10)
        , "n value for the factorial function")
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

