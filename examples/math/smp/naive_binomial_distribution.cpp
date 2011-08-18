////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <cmath>

#include <boost/cstdint.hpp>
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

using hpx::naming::id_type;

using hpx::applier::get_applier;

using hpx::actions::plain_result_action2;
using hpx::actions::plain_result_action4;

using hpx::lcos::eager_future;

using hpx::util::high_resolution_timer;

using hpx::init;
using hpx::finalize;

///////////////////////////////////////////////////////////////////////////////
double factorial(
    id_type const& prefix 
  , boost::int64_t n
);

typedef plain_result_action2<
    // result type
    double
    // arguments
  , id_type const&  
  , boost::int64_t
    // function
  , factorial
> factorial_action;

HPX_REGISTER_PLAIN_ACTION(factorial_action);

typedef eager_future<factorial_action> factorial_future;

///////////////////////////////////////////////////////////////////////////////
double binomial_distribution(
    id_type const& prefix 
  , boost::int64_t n
  , double p
  , boost::int64_t r
);

typedef plain_result_action4<
    // result type
    double
    // arguments
  , id_type const&  
  , boost::int64_t
  , double 
  , boost::int64_t
    // function
  , binomial_distribution
> binomial_distribution_action;

HPX_REGISTER_PLAIN_ACTION(binomial_distribution_action);

typedef eager_future<binomial_distribution_action> binomial_distribution_future;

///////////////////////////////////////////////////////////////////////////////
double factorial(
    id_type const& prefix
  , boost::int64_t n
) {
    std::cout << "factorial(" << n << ")\n";

    if (n <= 0)
        return 1;

    else
    {
        factorial_future n1(prefix, prefix, n - 1); 
        return n * n1.get();
    }
}

///////////////////////////////////////////////////////////////////////////////
double binomial_distribution(
    id_type const& prefix
  , boost::int64_t n
  , double p
  , boost::int64_t r
) {
    factorial_future fn(prefix, prefix, n)
                   , fnr(prefix, prefix, n -r)
                   , fr(prefix, prefix, r);    

    double x = std::pow(p, r) * std::pow(1 - p, n - r);
    double d = fnr.get() * fr.get();
    
    return (fn.get() / d) * x;
} 

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        boost::int64_t n = vm["n-value"].as<boost::int64_t>();
        double p = vm["p-value"].as<double>();
        boost::int64_t r = vm["r-value"].as<boost::int64_t>();

        const id_type prefix = get_applier().get_runtime_support_gid();

        high_resolution_timer t;

        binomial_distribution_future f(prefix, prefix, n, p, r);

        double x = f.get();

        double elapsed = t.elapsed();

        std::cout
            << ( boost::format("binomial_distribution(%1%, %2%, %3%) == %4%\n"
                               "elapsed time == %5%\n")
               % n % p % r % x % elapsed);
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
        , value<boost::int64_t>()->default_value(5)
        , "n value for the binomial distribution") 

        ( "p-value"
        , value<double>()->default_value(10.5)
        , "p value for the binomial distribution") 

        ( "r-value"
        , value<boost::int64_t>()->default_value(7)
        , "r value for the binomial distribution") 
        ;

    // Initialize and run HPX
    return init(desc_commandline, argc, argv);
}

