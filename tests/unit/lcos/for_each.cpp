//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/for_each.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>


#include <boost/lexical_cast.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/range/irange.hpp>

int twice(int i)
{
    return i * 2;
}

using boost::program_options::variables_map;
using boost::program_options::options_description;

int hpx_main(variables_map&)
{
    {
        std::vector<int> v(10);
        std::vector<int> w;
        boost::copy(boost::irange(0, 10), v.begin());

        hpx::lcos::wait(
            hpx::for_each(
                v
              , HPX_STD_BIND(twice, HPX_STD_PLACEHOLDERS::_1)
            )
          , w
        );

        int ref = 0;
        BOOST_FOREACH(int i, v)
        {
            HPX_TEST(i == ref++);
        }
        ref = 0;
        BOOST_FOREACH(int i, w)
        {
            HPX_TEST(i == (ref++)*2);

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

    // We force this test to use several threads by default.
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::hardware_concurrency());

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv, cfg);
}
