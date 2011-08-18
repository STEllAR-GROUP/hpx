//  Copyright (c) 2003-2007 Christopher M. Kohlhoff
//  Copyright (c) 2007-2010 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/agas/client/legacy/user_agent.hpp>
#include <hpx/runtime/agas/database/backend/stdmap.hpp>
#include <hpx/util/lightweight_test.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using boost::fusion::at_c;

using hpx::naming::locality;
using hpx::naming::address;
using hpx::naming::gid_type;
using hpx::naming::id_type;

using hpx::applier::get_applier;

using hpx::agas::legacy::user_agent;
using hpx::agas::user_primary_namespace;
using hpx::agas::user_component_namespace;
using hpx::agas::user_symbol_namespace;

using hpx::init;
using hpx::finalize;

using hpx::util::report_errors;

typedef user_agent<
    hpx::agas::tag::database::stdmap
> legacy_agent_type;

typedef user_primary_namespace<
    hpx::agas::tag::database::stdmap, hpx::agas::tag::network::tcpip
> primary_namespace_type;

typedef user_component_namespace<
    hpx::agas::tag::database::stdmap
> component_namespace_type;

typedef user_symbol_namespace<
    hpx::agas::tag::database::stdmap
> symbol_namespace_type;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    std::size_t iterations = 0;

    if (vm.count("iterations"))
        iterations = vm["iterations"].as<std::size_t>();
   
    legacy_agent_type agent;

    gid_type last_lowerid;
    locality here("127.0.0.1", 40000);

    for (std::size_t i = 0; i < iterations; ++i)
    {
        // Retrieve the id prefix of this site.
        gid_type prefix1;
        agent.get_prefix(here, prefix1);
        if (i == 0)
          last_lowerid = prefix1;
        HPX_TEST(prefix1 != 0);
 
        std::vector<gid_type> prefixes;
        agent.get_prefixes(prefixes);
        HPX_TEST((i != 0) ? (2 == prefixes.size()) : (1 == prefixes.size()));
        HPX_TEST_EQ(prefixes.back(), prefix1);
    
         // Identical sites should get same prefix.
        gid_type prefix2;
        agent.get_prefix(here, prefix2);
        HPX_TEST_EQ(prefix2, prefix1);
    
        // Different sites should get different prefix.
        gid_type prefix3;
        agent.get_prefix(locality("1.1.1.1", 1), prefix3);
        HPX_TEST(prefix3 != prefix2); 
   
        prefixes.clear(); 
        agent.get_prefixes(prefixes);
        HPX_TEST_EQ(2U, prefixes.size());
        HPX_TEST_EQ(prefixes.front(), prefix3);
    
        gid_type prefix4;
        agent.get_prefix(locality("1.1.1.1", 1), prefix4);
        HPX_TEST_EQ(prefix3, prefix4);   
    
        // test get_id_range
        gid_type lower1, upper1;
        HPX_TEST(agent.get_id_range(here, 1024, lower1, upper1));
        if (0 != i)
            HPX_TEST_EQ(last_lowerid + 1, lower1);   
 
        gid_type lower2, upper2;
        HPX_TEST(agent.get_id_range(here, 1024, lower2, upper2));
        HPX_TEST_EQ(upper1 + 1, lower2);   
        last_lowerid = upper2;
         
        // bind an arbitrary address
        HPX_TEST(agent.bind(gid_type(1), address(here, 1, 2)));
        
        // associate this id with a namespace name
        HPX_TEST(agent.registerid("/test/foo/1", gid_type(1)));
        
        // resolve this address
        address addr;
        HPX_TEST(agent.resolve(gid_type(1), addr, false));
        HPX_TEST_EQ(addr, address(here, 1, 2));
    
        // try to resolve a non-existing address
        HPX_TEST(!agent.resolve(gid_type(2), addr, false));
    
        // check association of the namespace name
        gid_type id;
        HPX_TEST(agent.queryid("/test/foo/1", id));
        HPX_TEST_EQ(id, gid_type(1));
        
        // rebind the id above to a new address
        HPX_TEST(!agent.bind(gid_type(1), address(here, 1, 3)));
    
        // re-associate this id with a namespace name
        HPX_TEST(!agent.registerid("/test/foo/1", gid_type(2)));
    
        // resolve it again
        HPX_TEST(agent.resolve(gid_type(1), addr, false));
        HPX_TEST_EQ(addr, address(here, 1, 3));
    
        // re-check association of the namespace name
        HPX_TEST(agent.queryid("/test/foo/1", id));
        HPX_TEST_EQ(id, gid_type(2));
    
        // unbind the address
        HPX_TEST(agent.unbind(gid_type(1)));
    
        // remove association
        HPX_TEST(agent.unregisterid("/test/foo/1"));
        
        // resolve should fail now
        HPX_TEST(!agent.resolve(gid_type(1), addr, false));
    
        // association of the namespace name should fail now
        HPX_TEST(!agent.queryid("/test/foo/1", id));
    
        // repeated unbind should fail
        HPX_TEST(!agent.unbind(gid_type(1)));
    
        // repeated remove association should fail
        HPX_TEST(!agent.unregisterid("/test/foo/1"));
        
        // test bind_range/unbind_range API
        HPX_TEST(agent.bind_range(gid_type(3), 20, address(here, 1, 2), 10));
    
        HPX_TEST(agent.resolve(gid_type(3), addr, false));
        HPX_TEST_EQ(addr, address(here, 1, 2));
        
        HPX_TEST(agent.resolve(gid_type(6), addr, false));
        HPX_TEST_EQ(addr, address(here, 1, 32));
    
        HPX_TEST(agent.resolve(gid_type(22), addr, false));
        HPX_TEST_EQ(addr, address(here, 1, 192));
        
        HPX_TEST(!agent.resolve(gid_type(23), addr, false));
    
        HPX_TEST(agent.unbind_range(gid_type(3), 20));
    }

    // initiate shutdown of the runtime system
    finalize(5.0);
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("iterations", value<std::size_t>()->default_value(1 << 6), 
            "the number of times to repeat the test") 
        ;

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(init(desc_commandline, argc, argv), 0,
      "HPX main exited with non-zero status");
    return report_errors();
}

