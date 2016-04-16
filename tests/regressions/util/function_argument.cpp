//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>

#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <vector>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::naming::id_type;
using hpx::async;

///////////////////////////////////////////////////////////////////////////////
bool invoked_f = false;
bool invoked_g = false;

bool f(hpx::util::function<bool()> func)
{
    HPX_TEST(!invoked_f);
    invoked_f = true;

    invoked_g = false;
    bool result = func();
    HPX_TEST(invoked_g);

    HPX_TEST(result);
    return result;
}

HPX_PLAIN_ACTION(f, f_action)

///////////////////////////////////////////////////////////////////////////////
struct g
{
    bool operator()()
    {
        HPX_TEST(!invoked_g);
        invoked_g = true;

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

        for (id_type const& prefix : prefixes)
        {
            invoked_f = false;
            HPX_TEST(async<f_action>(prefix, f).get());
            HPX_TEST(invoked_f);
        }
    }

    hpx::finalize();
    return hpx::util::report_errors();
}

int main(int argc, char **argv)
{
    options_description
        cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    return hpx::init(cmdline, argc, argv);
}
