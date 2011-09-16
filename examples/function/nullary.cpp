///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/include/iostreams.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/function.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/lcos/eager_future.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::actions::plain_result_action0;
using hpx::actions::plain_action1;
using hpx::actions::function;

using hpx::lcos::eager_future;
using hpx::lcos::future_value;

using hpx::naming::id_type;

using hpx::applier::get_applier;

using hpx::cout;
using hpx::endl;

using hpx::init;
using hpx::finalize;

///////////////////////////////////////////////////////////////////////////////
bool true_() { return true; }

typedef plain_result_action0<bool, true_> true_action;

HPX_REGISTER_PLAIN_ACTION(true_action);

///////////////////////////////////////////////////////////////////////////////
bool false_() { return false; }

typedef plain_result_action0<bool, false_> false_action;

HPX_REGISTER_PLAIN_ACTION(false_action);

///////////////////////////////////////////////////////////////////////////////
void print(function<bool()> f)
{ 
    cout << "f() == " << f() << endl; 
} 

typedef plain_action1<function<bool()>, print> print_action;

HPX_REGISTER_PLAIN_ACTION(print_action);

typedef eager_future<print_action> print_future;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        std::list<future_value<void> > futures;
        std::vector<id_type> prefixes;

        // Get a list of the global addresses of all machines in our system
        get_applier().get_prefixes(prefixes);

        // Start two print futures on each machine
        BOOST_FOREACH(id_type const& node, prefixes)
        {
            futures.push_back
                (print_future(node, function<bool()>(new true_action)));
            futures.push_back
                (print_future(node, function<bool()>(new false_action)));
        }

        // Wait for all IO to finish
        BOOST_FOREACH(future_value<void> const& f, futures)
        { f.get(); } 
    }

    // Shutdown all nodes
    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

