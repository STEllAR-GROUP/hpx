////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// Naive SMP version implemented with eager_futures and arbitrary size integers.

#include <iostream>

#include <boost/cstdint.hpp>
#include <boost/bigint.hpp>
#include <boost/bigint/serialize.hpp>
#include <boost/format.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/lcos/eager_future.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using boost::bigint;

using hpx::naming::id_type;

using hpx::applier::get_applier;

using hpx::actions::plain_result_action3;

using hpx::lcos::eager_future;

using hpx::util::high_resolution_timer;

using hpx::init;
using hpx::finalize;

///////////////////////////////////////////////////////////////////////////////
bigint ackermann_peter(
    id_type const& prefix 
  , boost::uint64_t m
  , bigint const& n
);

typedef plain_result_action3<
    // result type
    bigint
    // arguments
  , id_type const&  
  , boost::uint64_t
  , bigint const& 
    // function
  , ackermann_peter
> ackermann_peter_action;

HPX_REGISTER_PLAIN_ACTION(ackermann_peter_action);

typedef eager_future<ackermann_peter_action> ackermann_peter_future;

///////////////////////////////////////////////////////////////////////////////
bigint ackermann_peter(
    id_type const& prefix 
  , boost::uint64_t m
  , bigint const& n
) {
    if (0 == m)
        return n + 1;

    else
    {
        if (n == 0)
        {
            ackermann_peter_future m1(prefix, prefix, m - 1, 1); 
            return m1.get();
        }

        else
        {
            ackermann_peter_future m2(prefix, prefix, m, n - 1);
            ackermann_peter_future m1(prefix, prefix, m - 1, m2.get());
            return m1.get();
        }
    } 
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        boost::uint64_t m = vm["m-value"].as<boost::uint64_t>()
                      , n = vm["n-value"].as<boost::uint64_t>();

        const id_type prefix = get_applier().get_runtime_support_gid();

        high_resolution_timer t;

        ackermann_peter_future f(prefix, prefix, m, bigint(n));

        bigint r = f.get();

        double elapsed = t.elapsed();

        std::cout
            << ( boost::format("ackermann_peter(%1%, %2%) == %3%\n"
                               "elapsed time == %4%\n")
               % m % n % r % elapsed);
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
        ( "m-value"
        , value<boost::uint64_t>()->default_value(2)
        , "m value for Ackermann-Peter function") 

        ( "n-value"
        , value<boost::uint64_t>()->default_value(2)
        , "n value for the Ackermann-Peter function") 
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

