//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/runtime/agas/interface.hpp>

#include <boost/chrono.hpp>

#include <string>
#include <vector>

#include <tests/unit/agas/components/simple_refcnt_checker.hpp>
#include <tests/unit/agas/components/managed_refcnt_checker.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;
using hpx::find_here;

using boost::chrono::milliseconds;

using hpx::naming::id_type;
using hpx::naming::gid_type;
using hpx::naming::get_management_type_name;
using hpx::naming::detail::get_stripped_gid;

using hpx::agas::register_name_sync;
using hpx::agas::unregister_name_sync;

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
        /// AGAS reference-counting test 9 (from #126):
        ///
        ///     Create a component locally, and register its credit-stripped
        ///     raw gid with a symbolic name. Then, let all references to the
        ///     component go out of scope. The component should be destroyed.
        ///     Finally, unregister the symbolic name. Unregistering the
        ///     symbolic name should not cause any errors.

        char const name[] = "/tests(refcnt_checker#9)";

        Client monitor(find_here());

        cout << "id: " << monitor.get_id() << " "
             << get_management_type_name
                    (monitor.get_id().get_management_type()) << "\n"
             << flush;

        // Associate a symbolic name with the object. The symbol namespace
        // should not reference-count the name, as the GID we're passing has
        // no credits.
        gid_type raw_gid = get_stripped_gid(monitor.get_raw_gid());
        HPX_TEST_EQ(true, register_name_sync(name, raw_gid));

        {
            // Detach the reference.
            id_type id = monitor.detach().get();

            // The component should still be alive.
            HPX_TEST_EQ(false, monitor.is_ready(milliseconds(delay)));

            // let id go out of scope. id was the last reference to the
            // component
        }

        // The component should not be alive anymore, as the symbolic binding
        // does not hold a reference to it.
        HPX_TEST_EQ(true, monitor.is_ready(milliseconds(delay)));

        // Remove the symbolic name.
        HPX_TEST_EQ(raw_gid, unregister_name_sync(name).get_gid());

        // The component should be out of scope now.
        HPX_TEST_EQ(true, monitor.is_ready(milliseconds(delay)));
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
        , value<boost::uint64_t>()->default_value(500)
        , "number of milliseconds to wait for object destruction")
        ;

    // We need to explicitly enable the test components used by this test.
    std::vector<std::string> const cfg = {
        "hpx.components.simple_refcnt_checker.enabled! = 1",
        "hpx.components.managed_refcnt_checker.enabled! = 1"
    };

    // Initialize and run HPX.
    return init(cmdline, argc, argv, cfg);
}

