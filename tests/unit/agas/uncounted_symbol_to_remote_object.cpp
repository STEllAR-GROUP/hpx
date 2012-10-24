//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/agas/interface.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/assign/std/vector.hpp>

#include <tests/unit/agas/components/simple_refcnt_checker.hpp>
#include <tests/unit/agas/components/managed_refcnt_checker.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;
using hpx::find_here;

using boost::posix_time::milliseconds;

using hpx::naming::id_type;
using hpx::naming::gid_type;
using hpx::naming::get_management_type_name;
using hpx::naming::strip_credit_from_gid;

using hpx::components::component_type;
using hpx::components::get_component_type;

using hpx::applier::get_applier;

using hpx::agas::register_name;
using hpx::agas::unregister_name;
using hpx::agas::garbage_collect;

using hpx::test::simple_refcnt_monitor;
using hpx::test::managed_refcnt_monitor;

using hpx::util::report_errors;

using hpx::cout;
using hpx::flush;

///////////////////////////////////////////////////////////////////////////////
template <
    typename Client
>
void hpx_test_main(
    variables_map& vm
    )
{
    boost::uint64_t const delay = vm["delay"].as<boost::uint64_t>();

    {
        /// AGAS reference-counting test 10 (from #126):
        ///
        ///     Create a component remotely, and register it's credit-stripped
        ///     raw gid with a symbolic name. Then, let all references to the
        ///     component go out of scope. The component should be destroyed.
        ///     Finally, unregister the symbolic name. Unregistering the
        ///     symbolic name should not cause any errors.

        char const name[] = "/tests(refcnt_checker#10)";

        typedef typename Client::server_type server_type;

        component_type ctype = get_component_type<server_type>();
        std::vector<id_type> remote_localities = hpx::find_remote_localities(ctype);

        if (remote_localities.empty())
            throw std::logic_error("this test cannot be run on one locality");

        Client monitor(remote_localities[0]);

        cout << "id: " << monitor.get_gid() << " "
             << get_management_type_name
                    (monitor.get_gid().get_management_type()) << "\n"
             << flush;

        // Associate a symbolic name with the object. The symbol namespace
        // should not reference-count the name, as the GID we're passing has
        // no credits.
        gid_type raw_gid = strip_credit_from_gid(monitor.get_raw_gid());
        HPX_TEST_EQ(true, register_name(name, raw_gid));

        {
            // Detach the reference.
            monitor.detach();

            // The component should still be alive.
            HPX_TEST_EQ(false, monitor.ready(milliseconds(delay)));
        }

        // Flush pending reference counting operations.
        garbage_collect();

        // The component should be out of scope now.
        HPX_TEST_EQ(true, monitor.ready(milliseconds(delay)));

        // Remove the symbolic name.
        HPX_TEST_EQ(true, unregister_name(name));
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    {
        cout << std::string(80, '#') << "\n"
             << "simple component test\n"
             << std::string(80, '#') << "\n" << flush;

        hpx_test_main<simple_refcnt_monitor>(vm);

        cout << std::string(80, '#') << "\n"
             << "managed component test\n"
             << std::string(80, '#') << "\n" << flush;

        hpx_test_main<managed_refcnt_monitor>(vm);
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

    // We need to explicitly enable the test components used by this test.
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.components.simple_refcnt_checker.enabled = 1";
    cfg += "hpx.components.managed_refcnt_checker.enabled = 1";

    // Initialize and run HPX.
    return init(cmdline, argc, argv, cfg);
}

