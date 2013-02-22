//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/assign/std/vector.hpp>

#include <tests/unit/agas/components/simple_mobile_object.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::init;
using hpx::finalize;
using hpx::find_here;

using hpx::naming::gid_type;
using hpx::naming::get_agas_client;
using hpx::naming::detail::get_stripped_gid;
using hpx::naming::address;

using hpx::test::simple_mobile_object;

using hpx::util::report_errors;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    {
        simple_mobile_object a(hpx::find_here());
        simple_mobile_object b(hpx::find_here());

        gid_type a_id = get_stripped_gid(a.get_gid().get_gid());
        boost::uint64_t b_lva = b.get_lva();

        // Resolve a_id.
        address addr = get_agas_client().resolve(a_id);    

        ///////////////////////////////////////////////////////////////////////
        HPX_TEST_EQ(addr.address_, a.get_lva());
        HPX_SANITY_EQ(get_agas_client().resolve(a_id).address_, a.get_lva());

        ///////////////////////////////////////////////////////////////////////
        // Change a's GID to point to b. 

        // Rebind the GID.
        boost::uint64_t a_lva = addr.address_;
        addr.address_ = b_lva;
        HPX_TEST(get_agas_client().bind(a_id, addr));

        // Update our AGAS cache.
        get_agas_client().update_cache_entry(a_id, addr);

        ///////////////////////////////////////////////////////////////////////
        HPX_TEST_EQ(b_lva, a.get_lva());
        HPX_SANITY_EQ(get_agas_client().resolve(a_id).address_, a.get_lva());

        ///////////////////////////////////////////////////////////////////////
        // Now we restore the original bindings to prevent a double free. 

        // Rebind the GID.
        addr.address_ = a_lva;
        HPX_TEST(get_agas_client().bind(a_id, addr));

        // Update our AGAS cache.
        get_agas_client().update_cache_entry(a_id, addr);

        ///////////////////////////////////////////////////////////////////////
        HPX_TEST_EQ(get_agas_client().resolve(a_id).address_, a_lva);
        HPX_SANITY_EQ(get_agas_client().resolve(a_id).address_, a.get_lva());
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

    // We need to explicitly enable the test components used by this test.
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.components.simple_mobile_object.enabled! = 1";

    // Initialize and run HPX.
    return init(cmdline, argc, argv, cfg);
}

