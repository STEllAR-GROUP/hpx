//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/include/lcos.hpp>

#include <tests/unit/agas/components/simple_refcnt_checker.hpp>
#include <tests/unit/agas/components/managed_refcnt_checker.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;
using hpx::find_here;

using hpx::naming::id_type;
using hpx::naming::get_management_type_name;
using hpx::naming::get_locality_id_from_id;
using hpx::naming::detail::split_credits_for_gid;
using hpx::naming::detail::get_credit_from_gid;

using hpx::components::component_type;
using hpx::components::get_component_type;

using hpx::applier::get_applier;

using hpx::async;

using hpx::test::simple_object;
using hpx::test::managed_object;

using hpx::util::report_errors;

using hpx::cout;
using hpx::flush;
using hpx::find_here;

///////////////////////////////////////////////////////////////////////////////
// helper functions
inline boost::uint64_t get_credit(id_type const& id)
{
    return hpx::naming::detail::get_credit_from_gid(id.get_gid());
}

inline id_type split_credits(id_type const& id)
{
    return id_type(
        split_credits_for_gid(const_cast<id_type&>(id).get_gid()),
        id_type::managed);
}

///////////////////////////////////////////////////////////////////////////////
template <
    typename Client
>
void hpx_test_main(
    variables_map& vm
    )
{
    boost::uint64_t const hpx_globalcredit_initial = HPX_GLOBALCREDIT_INITIAL;

    // HPX_GLOBALCREDIT_INITIAL should be a power of 2
    boost::uint16_t log2_initial_credit =
        hpx::naming::detail::log2(hpx_globalcredit_initial);
    boost::uint64_t restored_initial_credits =
        hpx::naming::detail::power2(log2_initial_credit);
    HPX_TEST_EQ(restored_initial_credits, hpx_globalcredit_initial);

    {
        Client object(find_here());

        id_type g0 = split_credits(object.get_gid());

        HPX_TEST_EQ(get_credit(object.get_gid()), hpx_globalcredit_initial/2);
        HPX_TEST_EQ(get_credit(g0), hpx_globalcredit_initial/2);

        id_type g1 = split_credits(object.get_gid());

        HPX_TEST_EQ(get_credit(object.get_gid()), hpx_globalcredit_initial/4);
        HPX_TEST_EQ(get_credit(g1), hpx_globalcredit_initial/4);

        cout << "  " << object.get_gid() << " : "
                     << get_credit(object.get_gid()) << "\n"
             << "  " << g0 << " : "
                     << get_credit(g0) << "\n"
             << "  " << g1 << " : "
                     << get_credit(g1) << "\n" << flush;
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

        hpx_test_main<simple_object>(vm);

        cout << std::string(80, '#') << "\n"
             << "managed component test\n"
             << std::string(80, '#') << "\n" << flush;

        hpx_test_main<managed_object>(vm);
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

