//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>

#include <ctime>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::init;
using hpx::finalize;

using hpx::no_success;

using hpx::applier::apply;
using hpx::applier::get_applier;
using hpx::applier::get_prefix_id;

using hpx::actions::plain_action0;

using hpx::naming::id_type;

using hpx::threads::pending;
using hpx::threads::suspended;
using hpx::threads::thread_state_ex_enum;
using hpx::threads::wait_timeout;
using hpx::threads::get_self_id;
using hpx::threads::get_self;
using hpx::threads::set_thread_state;

using boost::posix_time::milliseconds;

///////////////////////////////////////////////////////////////////////////////
void raise_exception()
{
    std::cout << (boost::format("raising exception on %d\n") % get_prefix_id())
              << std::flush;
    HPX_THROW_EXCEPTION(no_success, "raise_exception", "unhandled exception"); 
}

typedef plain_action0<raise_exception> raise_exception_action;

HPX_REGISTER_PLAIN_ACTION(raise_exception_action);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        std::vector<id_type> localities;
        get_applier().get_remote_prefixes(localities);

        if (localities.empty())
            raise_exception();
 
        boost::mt19937 rng(std::time(NULL));
        boost::uniform_int<std::size_t>
            locality_range(0, localities.size() - 1);

        apply<raise_exception_action>(localities[locality_range(rng)]);

        thread_state_ex_enum statex = wait_timeout;

        while (statex == wait_timeout)
        {
            // Schedule a wakeup in 500 milliseconds.
            set_thread_state(get_self_id(), milliseconds(500), pending);

            // Suspend this pxthread.
            statex = get_self().yield(suspended);
        }
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

