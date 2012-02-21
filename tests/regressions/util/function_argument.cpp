//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>

#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/function.hpp>

#include <boost/detail/lightweight_test.hpp>
#include <boost/foreach.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::actions::plain_result_action1;
using hpx::naming::id_type;
using hpx::lcos::async;

///////////////////////////////////////////////////////////////////////////////
bool invoked_f = false;
bool invoked_g = false;

bool f(hpx::util::function<bool()> func)
{
    BOOST_TEST(!invoked_f);
    invoked_f = true;

    invoked_g = false;
    bool result = func();
    BOOST_TEST(invoked_g);

    BOOST_TEST(result);
    return result;
}

typedef plain_result_action1<bool, hpx::util::function<bool()>, &f> f_action;

HPX_REGISTER_PLAIN_ACTION(f_action);

///////////////////////////////////////////////////////////////////////////////
struct g
{
    bool operator()()
    {
        BOOST_TEST(!invoked_g);
        invoked_g = true;

        hpx::cout << "Hello World\n" << hpx::flush;
        return true;
    }

    // dummy serialization functionality
    template <typename Archive>
    void serialize(Archive &, unsigned) {}
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map &)
{
    {
        hpx::util::function<bool()> f = g();
        std::vector<id_type> prefixes = hpx::find_all_localities();

        BOOST_FOREACH(id_type const & prefix, prefixes)
        {
            invoked_f = false;
            BOOST_TEST(async<f_action>(prefix, f).get());
            BOOST_TEST(invoked_f);
        }
    }

    hpx::finalize();
    return boost::report_errors();
}

int main(int argc, char **argv)
{
    options_description
        cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    return hpx::init(cmdline, argc, argv);
}
