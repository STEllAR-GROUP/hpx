//  Copyright (c) 2011 Bryce Adelstein-Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>

#include <tests/correctness/agas/components/simple_refcnt_checker.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;

using boost::posix_time::milliseconds;

using hpx::naming::id_type;

using hpx::components::component_type;
using hpx::components::get_component_type;

using hpx::applier::get_applier;

using hpx::test::simple_refcnt_checker;

using hpx::util::report_errors;

using hpx::cout;
using hpx::flush;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    const boost::uint64_t delay = vm["delay"].as<boost::uint64_t>();

    {
        /// AGAS reference-counting test 2 (from #126):
        ///
        ///     Create a component remotely and let all references to it go out
        ///     of scope. The component should be deleted.

        std::vector<id_type> remote_localities;

        typedef hpx::test::server::simple_refcnt_checker server_type;

        component_type ctype = get_component_type<server_type>();

        if (!get_applier().get_remote_prefixes(remote_localities, ctype))
            throw std::logic_error("this test cannot be run on one locality");

        simple_refcnt_checker monitor(remote_localities[0]);

        cout << "id: " << monitor.get_gid() << "\n" << flush;

        {
            // Detach the reference.
            id_type id = monitor.detach();

            // The component should still be alive.
            HPX_TEST_EQ(false, monitor.ready(milliseconds(delay))); 
        }

        // The component should be out of scope now.
        HPX_TEST_EQ(true, monitor.ready(milliseconds(delay))); 
    }

    finalize();
    return report_errors();
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
        ( "delay"
        , value<boost::uint64_t>()->default_value(1000)
        , "number of milliseconds to wait for object destruction") 
        ;

    // Initialize and run HPX.
    return init(cmdline, argc, argv);
}

