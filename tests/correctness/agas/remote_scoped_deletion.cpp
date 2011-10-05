//  Copyright (c) 2011 Bryce Adelstein-Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>

#include <tests/correctness/agas/components/managed_refcnt_checker.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;
using hpx::find_here;

using boost::posix_time::milliseconds;

using hpx::naming::id_type;

using hpx::components::component_type;
using hpx::components::get_component_type;

using hpx::applier::get_applier;

using hpx::test::managed_refcnt_checker;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    {
        // AGAS reference-counting test 1 (from #126): create a component
        // remotely and let all references to it go out of scope. The component
        // instance should be deleted.

        std::vector<id_type> remote_localities;

        typedef hpx::test::server::managed_refcnt_checker server_type;

        component_type ctype = get_component_type<server_type>();

        if (!get_applier().get_remote_prefixes(remote_localities, ctype))
            throw std::logic_error("this test cannot be run on one locality");

        managed_refcnt_checker monitor(remote_localities[0]);

        {
            id_type id = monitor.detach();

            // The component should still be alive.
            HPX_TEST_EQ(false, monitor.ready(milliseconds(1000))); 
        }

        // The component should be out of scope now.
        HPX_TEST_EQ(true, monitor.ready(milliseconds(1000))); 
    }

    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(
    int argc
  , char* argv[]
    )
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return init(cmdline, argc, argv);
}

