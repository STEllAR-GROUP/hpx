//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/iostream.hpp>
#include <hpx/modules/testing.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "components/managed_refcnt_checker.hpp"
#include "components/simple_refcnt_checker.hpp"

using hpx::program_options::options_description;
using hpx::program_options::value;
using hpx::program_options::variables_map;

using hpx::finalize;
using hpx::find_here;
using hpx::init;

using hpx::id_type;
using hpx::naming::get_locality_id_from_id;
using hpx::naming::get_management_type_name;
using hpx::naming::detail::get_credit_from_gid;
using hpx::naming::detail::split_credits_for_gid;

using hpx::components::component_type;
using hpx::components::get_component_type;

using hpx::applier::get_applier;

using hpx::async;

using hpx::test::managed_object;
using hpx::test::simple_object;

using hpx::util::report_errors;

using hpx::cout;
using hpx::find_here;

///////////////////////////////////////////////////////////////////////////////
// helper functions
inline std::int64_t get_credit(id_type const& id)
{
    return hpx::naming::detail::get_credit_from_gid(id.get_gid());
}

inline id_type split_credits(id_type const& id)
{
    return id_type(split_credits_for_gid(const_cast<id_type&>(id).get_gid()),
        id_type::management_type::managed);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Client>
void hpx_test_main(variables_map&, hpx::id_type const& locality)
{
    std::int64_t const hpx_globalcredit_initial =
        static_cast<std::int64_t>(HPX_GLOBALCREDIT_INITIAL);

    // HPX_GLOBALCREDIT_INITIAL should be a power of 2
    std::int16_t log2_initial_credit =
        hpx::naming::detail::log2(hpx_globalcredit_initial);
    std::int64_t restored_initial_credits =
        hpx::naming::detail::power2(log2_initial_credit);
    HPX_TEST_EQ(restored_initial_credits, hpx_globalcredit_initial);

    {
        Client object(locality);

        id_type g0 = split_credits(object.get_id());

        HPX_TEST_EQ(get_credit(object.get_id()), hpx_globalcredit_initial / 2);
        HPX_TEST_EQ(get_credit(g0), hpx_globalcredit_initial / 2);

        id_type g1 = split_credits(object.get_id());

        HPX_TEST_EQ(get_credit(object.get_id()), hpx_globalcredit_initial / 4);
        HPX_TEST_EQ(get_credit(g1), hpx_globalcredit_initial / 4);

        id_type g2 = split_credits(object.get_id());

        HPX_TEST_EQ(get_credit(object.get_id()), hpx_globalcredit_initial / 8);
        HPX_TEST_EQ(get_credit(g2), hpx_globalcredit_initial / 8);

        id_type g3 = split_credits(object.get_id());

        HPX_TEST_EQ(get_credit(object.get_id()), hpx_globalcredit_initial / 16);
        HPX_TEST_EQ(get_credit(g3), hpx_globalcredit_initial / 16);

        id_type g4 = split_credits(object.get_id());

        HPX_TEST_EQ(get_credit(object.get_id()), hpx_globalcredit_initial / 32);
        HPX_TEST_EQ(get_credit(g4), hpx_globalcredit_initial / 32);

        id_type g5 = split_credits(object.get_id());

        HPX_TEST_EQ(get_credit(object.get_id()), hpx_globalcredit_initial / 64);
        HPX_TEST_EQ(get_credit(g5), hpx_globalcredit_initial / 64);

        cout << "  " << object.get_id() << " : " << get_credit(object.get_id())
             << "\n"
             << "  " << g0 << " : " << get_credit(g0) << "\n"
             << "  " << g1 << " : " << get_credit(g1) << "\n"
             << "  " << g2 << " : " << get_credit(g2) << "\n"
             << "  " << g3 << " : " << get_credit(g3) << "\n"
             << "  " << g4 << " : " << get_credit(g4) << "\n"
             << "  " << g5 << " : " << get_credit(g5) << "\n"
             << std::flush;
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    for (hpx::id_type const& l : hpx::find_all_localities())
    {
        cout << std::string(80, '#') << "\n"
             << "simple component test: " << l << "\n"
             << std::string(80, '#') << "\n"
             << std::flush;

        hpx_test_main<simple_object>(vm, l);

        cout << std::string(80, '#') << "\n"
             << "managed component test: " << l << "\n"
             << std::string(80, '#') << "\n"
             << std::flush;

        hpx_test_main<managed_object>(vm, l);
    }

    finalize();
    return report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()("delay", value<std::uint64_t>()->default_value(1000),
        "number of milliseconds to wait for object destruction");

    // We need to explicitly enable the test components used by this test.
    std::vector<std::string> const cfg = {
        "hpx.components.simple_refcnt_checker.enabled! = 1",
        "hpx.components.managed_refcnt_checker.enabled! = 1"};

    // Initialize and run HPX.
    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
#endif
