///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <iomanip>
#include <cmath>

#include <boost/math/constants/constants.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/function.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/lcos/eager_future.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::actions::plain_result_action1;
using hpx::actions::plain_action3;
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
double sin_(double x) { return std::sin(x); }

typedef plain_result_action1<double, double, sin_> sin_action;

HPX_REGISTER_PLAIN_ACTION(sin_action);

///////////////////////////////////////////////////////////////////////////////
double cos_(double x) { return std::cos(x); }

typedef plain_result_action1<double, double, cos_> cos_action;

HPX_REGISTER_PLAIN_ACTION(cos_action);

///////////////////////////////////////////////////////////////////////////////
double tan_(double x) { return std::tan(x); }

typedef plain_result_action1<double, double, tan_> tan_action;

HPX_REGISTER_PLAIN_ACTION(tan_action);

///////////////////////////////////////////////////////////////////////////////
void print(std::string const& name, function<double(double)> f, double x)
{
    cout << std::setprecision(12)
         << name << "(" << x << ") == " << f(x) << endl;
} 

typedef plain_action3<
    // arguments
    std::string const&
  , function<double(double)>
  , double
    // function
  , print
> print_action;

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

        const double pi = boost::math::constants::pi<double>();

        function<double(double)> sin_f(new sin_action);
        function<double(double)> cos_f(new cos_action);
        function<double(double)> tan_f(new tan_action);

        // Start two print futures on each machine
        BOOST_FOREACH(id_type const& node, prefixes)
        {
            futures.push_back(print_future(node, "sin", sin_f, 0));
            futures.push_back(print_future(node, "sin", sin_f, pi));
            futures.push_back(print_future(node, "cos", cos_f, 0));
            futures.push_back(print_future(node, "cos", cos_f, pi));
            futures.push_back(print_future(node, "tan", tan_f, 0));
            futures.push_back(print_future(node, "tan", tan_f, pi));
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

