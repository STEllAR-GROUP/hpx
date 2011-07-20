//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <ctime>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/exception.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::init;
using hpx::finalize;
using hpx::no_success;

using hpx::applier::get_applier;

using hpx::actions::plain_action0;

using hpx::lcos::eager_future;

using hpx::naming::gid_type;

///////////////////////////////////////////////////////////////////////////////
void raise_exception()
{
    HPX_THROW_EXCEPTION(no_success, "raise_exception", "unhandled exception"); 
}

typedef plain_action0<raise_exception> raise_exception_action;

HPX_REGISTER_PLAIN_ACTION(raise_exception_action);

typedef eager_future<raise_exception_action> raise_exception_future;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        std::vector<gid_type> localities;
        get_applier().get_agas_client().get_prefixes(localities);
         
        boost::mt19937 rng(std::time(NULL));
        boost::uniform_int<std::size_t>
            locality_range(0, localities.size() - 1);

        raise_exception_future f(localities[locality_range(rng)]);

        f.get();
    }

    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}

