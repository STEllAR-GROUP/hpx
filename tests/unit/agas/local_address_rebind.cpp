//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/iostream.hpp>
#include <hpx/modules/testing.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "components/simple_mobile_object.hpp"

using hpx::program_options::variables_map;
using hpx::program_options::options_description;

using hpx::init;
using hpx::finalize;
using hpx::find_here;

using hpx::naming::id_type;
using hpx::naming::gid_type;
using hpx::naming::get_agas_client;
using hpx::naming::detail::get_stripped_gid;
using hpx::naming::address;

using hpx::util::report_errors;

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        hpx::test::simple_mobile_object a =
            hpx::new_<hpx::test::server::simple_mobile_object>(hpx::find_here());
        hpx::test::simple_mobile_object b =
            hpx::new_<hpx::test::server::simple_mobile_object>(hpx::find_here());

        id_type a_id = a.get_id();
        gid_type a_gid = get_stripped_gid(a_id.get_gid());
        std::uint64_t b_lva = b.get_lva();

        // Resolve a_gid.
        address addr = hpx::agas::resolve(a_id).get();

        ///////////////////////////////////////////////////////////////////////
        HPX_TEST_EQ(addr.address_, a.get_lva());
        HPX_SANITY_EQ(hpx::agas::resolve(a_id).get().address_, a.get_lva());

        ///////////////////////////////////////////////////////////////////////
        // Change a's GID to point to b.

        // Rebind the GID.
        std::uint64_t a_lva = addr.address_;
        addr.address_ = b_lva;
        HPX_TEST(get_agas_client().bind_local(a_gid, addr));

        // Update our AGAS cache.
        get_agas_client().update_cache_entry(a_gid, addr);

        ///////////////////////////////////////////////////////////////////////
        HPX_TEST_EQ(b_lva, a.get_lva());
        HPX_SANITY_EQ(hpx::agas::resolve(a_id).get().address_, a.get_lva());

        ///////////////////////////////////////////////////////////////////////
        // Now we restore the original bindings to prevent a double free.

        // Rebind the GID.
        addr.address_ = a_lva;
        HPX_TEST(get_agas_client().bind_local(a_gid, addr));

        // Update our AGAS cache.
        get_agas_client().update_cache_entry(a_gid, addr);

        ///////////////////////////////////////////////////////////////////////
        HPX_TEST_EQ(hpx::agas::resolve(a_id).get().address_, a_lva);
        HPX_SANITY_EQ(hpx::agas::resolve(a_id).get().address_, a.get_lva());
    }

    finalize();
    return report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    // We need to explicitly enable the test components used by this test.
    std::vector<std::string> const cfg = {
        "hpx.components.simple_mobile_object.enabled! = 1"
    };

    // Initialize and run HPX.
    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
#endif
