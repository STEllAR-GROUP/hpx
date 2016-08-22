////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <boost/random.hpp>
#include <boost/format.hpp>

#include <functional>
#include <vector>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;

///////////////////////////////////////////////////////////////////////////////
// x to the p power
template <typename T>
inline T sign(T a)
{
    if (a > 0)
        return 1;
    else if (a < 0)
        return -1;
    else
        return 0;
}

template <typename T>
T ipow(T x, T p)
{
    T i = 1;
    for (T j = 1; j < p; j++) i *= x;
    return i;
}

template <typename T>
inline bool compare_real(T x, T y, T epsilon)
{
    if ((x + epsilon >= y) && (x - epsilon <= y))
        return true;
    else
        return false;
}

double null_function(
    boost::uint64_t seed
  , boost::uint64_t delay_iterations
    )
{
    boost::random::mt19937_64 prng(seed);

    boost::random::uniform_real_distribution<double> v(0, 1.);
    boost::random::uniform_smallint<boost::uint8_t> s(0, 1);

    double d = 0.;

    for (boost::uint64_t i = 0; i < delay_iterations; ++i)
    {
        double v0 = v(prng);
        double v1 = v(prng);

        if (compare_real(v1, 0.0, 1e-10))
            v1 = 1e-10;

        d += (s(prng) ? 1.0 : -1.0) * (v0 / v1);
    }

    return d;
}

///////////////////////////////////////////////////////////////////////////////
double null_tree(
    boost::uint64_t seed
  , boost::uint64_t depth
  , boost::uint64_t max_depth
  , boost::uint64_t children
  , boost::uint64_t delay_iterations
  , boost::uint32_t num_localities
    );

HPX_PLAIN_ACTION(null_tree, null_tree_action);

void null_callback(
    std::vector<double>& dd
  , boost::uint64_t j
  , hpx::future<double> f
    )
{
    dd[j] = f.get();
}

double null_tree(
    boost::uint64_t seed
  , boost::uint64_t depth
  , boost::uint64_t max_depth
  , boost::uint64_t children
  , boost::uint64_t delay_iterations
  , boost::uint32_t num_localities
    )
{
    if (depth == max_depth)
        return null_function(seed, delay_iterations);

    std::vector<double> dd;
    dd.resize(children, 0.);

    std::vector<hpx::future<void> > futures;
    futures.reserve(children);

    boost::uint64_t p = seed + ipow(depth, children);

    for (boost::uint64_t j = 0; j < children; ++j)
    {
        hpx::id_type const target
            = hpx::naming::get_id_from_locality_id((j + p) % num_localities);

        hpx::future<double> f =
            hpx::async<null_tree_action>(target, j + p, depth + 1
                                       , max_depth
                                       , children
                                       , delay_iterations
                                       , num_localities
                                        );

        futures.push_back(f.then
            (hpx::util::bind(&null_callback, std::ref(dd), j,
                hpx::util::placeholders::_1)));
    }

    null_function(seed, delay_iterations);

    hpx::wait_all(futures);

    double d = 0.;

    for (boost::uint64_t j = 0; j < children; ++j)
        d += dd[j];

    return d;
}

int hpx_main(
    variables_map& vm
    )
{
    {
        boost::uint64_t test_runs = vm["test-runs"].as<boost::uint64_t>();

        boost::uint64_t children = vm["children"].as<boost::uint64_t>();

        boost::uint64_t max_depth = vm["depth"].as<boost::uint64_t>() + 1;

        boost::uint64_t delay_iterations
            = vm["delay-iterations"].as<boost::uint64_t>();

        bool verbose = vm.count("verbose") != 0;

        boost::uint32_t num_localities =
            hpx::get_num_localities(hpx::launch::sync);

        hpx::id_type const here = hpx::find_here();

        double d = 0.;

        for ( boost::uint64_t i = 0
            ; (test_runs == 0) || (i < test_runs)
            ; ++i)
        {
            hpx::util::high_resolution_timer local_clock;

            d += hpx::async<null_tree_action>(here, 0, 1
                                            , max_depth
                                            , children
                                            , delay_iterations
                                            , num_localities
                                              ).get();

            if (verbose)
            {
                double step_speed = (1 / local_clock.elapsed());

                char const* fmt =
                    "%016u, %.7g,%|31t| %.7g%|41t| [steps/second]\n";

                std::cout << (boost::format(fmt) % i % d % step_speed)
                          << std::flush;

            }
        }
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(
    int argc
  , char* argv[]
    )
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ( "test-runs"
        , value<boost::uint64_t>()->default_value(10)
        , "number of times to repeat the test (0 == infinite)")

        ( "verbose"
        , "print state every iteration")

        ( "children"
        , value<boost::uint64_t>()->default_value(8)
        , "number of children each node has")

        ( "depth"
        , value<boost::uint64_t>()->default_value(3)
        , "depth of the tree structure")

        ( "delay-iterations"
        , value<boost::uint64_t>()->default_value(1000)
        , "number of iterations in the delay loop")
        ;

    // Initialize and run HPX.
    return hpx::init(cmdline, argc, argv);
}

