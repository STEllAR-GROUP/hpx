//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/lightweight_test.hpp>

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
        /// AGAS reference-counting test 3 (from #126):
        ///
        ///     Create two components locally, and have the second component
        ///     store a reference to the first component. Let the original
        ///     references to both components go out of scope. Both components
        ///     should be deleted.

        Client monitor0(find_here());
        Client monitor1(find_here());

        cout << "id0: " << monitor0.get_gid() << " "
             << get_management_type_name
                    (monitor0.get_gid().get_management_type()) << "\n"
             << "id1: " << monitor1.get_gid() << " "
             << get_management_type_name
                    (monitor1.get_gid().get_management_type()) << "\n"
             << flush;

        {
            // Have the second object store a reference to the first object.
            monitor1.take_reference(monitor0.get_gid());

            // Detach the references.
            id_type id1 = monitor0.detach().get();
            id_type id2 = monitor1.detach().get();

            // Both components should still be alive.
            HPX_TEST_EQ(false, monitor0.is_ready(milliseconds(delay)));
            HPX_TEST_EQ(false, monitor1.is_ready(milliseconds(delay)));
        }

        // Flush pending reference counting operations.
        garbage_collect();
        garbage_collect();

        // Both components should be out of scope now.
        HPX_TEST_EQ(true, monitor0.is_ready(milliseconds(delay)));
        HPX_TEST_EQ(true, monitor1.is_ready(milliseconds(delay)));
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
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.components.simple_refcnt_checker.enabled! = 1";
    cfg += "hpx.components.managed_refcnt_checker.enabled! = 1";

    // Initialize and run HPX.
    return init(cmdline, argc, argv, cfg);
}

