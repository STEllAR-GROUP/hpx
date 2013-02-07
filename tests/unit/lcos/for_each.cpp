//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/for_each.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/lightweight_test.hpp>


#include <boost/lexical_cast.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/range/irange.hpp>

boost::uint64_t delay = 0;

std::size_t twice(std::size_t i)
{
    double volatile d = 0.;
    for (boost::uint64_t ui = 0; ui < delay; ++ui)
        d += 1. / (2. * ui + 1.);
    return i * 2;
}

HPX_PLAIN_ACTION(twice, twice_action);

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;
using hpx::util::high_resolution_timer;

int hpx_main(variables_map & vm)
{
    {
        std::size_t n = vm["n"].as<std::size_t>();
        std::vector<std::size_t> v(n);
        std::vector<std::size_t> w;
        boost::copy(boost::irange(std::size_t(0), n), v.begin());

        high_resolution_timer t;
        twice_action act;
        hpx::lcos::wait(
            hpx::for_each(
                v
              , HPX_STD_BIND(act, hpx::find_here(), HPX_STD_PLACEHOLDERS::_1)
            )
          , w
        );

        std::cout << t.elapsed() << "\n";

        std::size_t ref = 0;
        BOOST_FOREACH(std::size_t i, v)
        {
            HPX_TEST_EQ(i, ref++);
        }
        ref = 0;
        BOOST_FOREACH(std::size_t i, w)
        {
            HPX_TEST_EQ(i, (ref++)*2);

        }
    }

    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");
    cmdline.add_options()
        ( "delay"
        , value<boost::uint64_t>(&delay)->default_value(0)
        , "number of iterations in the delay loop")
        ("n", value<std::size_t>()->default_value(10), 
            "the number of vector elements to iterate over") ;

    // We force this test to use several threads by default.
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency());

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv/*, cfg*/);
}
