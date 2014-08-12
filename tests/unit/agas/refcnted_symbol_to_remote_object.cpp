//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/runtime/agas/interface.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/chrono.hpp>

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
using hpx::naming::get_management_type_name;
using hpx::naming::get_locality_from_id;

using hpx::components::component_type;
using hpx::components::get_component_type;

using hpx::applier::get_applier;

using hpx::agas::register_name_sync;
using hpx::agas::unregister_name_sync;
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
        /// AGAS reference-counting test 8 (from #126):
        ///
        ///     Create a component remotely, and register a symbolic name for
        ///     it. Then, let all references to the component go out of scope.
        ///     The component should still be alive. Finally, unregister the
        ///     symbolic name. The component should be deleted after the
        ///     symbolic name is unregistered.

        char const name[] = "/tests(refcnt_checker#8)";

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

        hpx::naming::gid_type gid;

        // Associate a symbolic name with the object.
        HPX_TEST_EQ(true, register_name_sync(name, monitor.get_gid()));

        {
            // Detach the reference.
            id_type id = monitor.detach().get();

            // The component should still be alive.
            HPX_TEST_EQ(false, monitor.is_ready(milliseconds(delay)));

            gid = id.get_gid();

            // let id go out of scope
        }

        // The component should still be alive, as the symbolic binding holds
        // a reference to it.
        HPX_TEST_EQ(false, monitor.is_ready(milliseconds(delay)));

        // Remove the symbolic name. This should return the final credits
        // to AGAS.
        HPX_TEST_EQ(gid, unregister_name_sync(name).get_gid());

        // Flush pending reference counting operations.
        garbage_collect(remote_localities[0]);
        garbage_collect();
        garbage_collect(remote_localities[0]);
        garbage_collect();

        // The component should be destroyed.
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
        , value<boost::uint64_t>()->default_value(1000)
        , "number of milliseconds to wait for object destruction")
        ;

    // We need to explicitly enable the test components used by this test.
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.components.simple_refcnt_checker.enabled! = 1";
    cfg += "hpx.components.managed_refcnt_checker.enabled! = 1";

    // Initialize and run HPX.
    return init(cmdline, argc, argv, cfg);
}

