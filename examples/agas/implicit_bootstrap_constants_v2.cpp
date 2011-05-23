//  Copyright (c) 2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This is a proof-of-constant of the implicit constants used by the AGAS v2
// bootstrap sequence. Key concepts:
//
//  * Locality prefixes are assigned in a deterministic and reproducible
//    fashion.
//  * The AGAS locality is always the first locality to be registered with
//    AGAS.
//  * AGAS legacy client code is smart enough to transparently bootstrap the
//    AGAS components. Minimal, non-intrusive changes to the runtime class and
//    runtime subsystems are needed.
//  * The AGAS namespace components have static component ids which are known
//    at compile time and do not need to be assigned by AGAS.
//  * The AGAS operations used during simple component registration by the
//    component subsystem are the only operations which need to be fulfilled
//    by the client during the bootstrap process.
//  * The bootstrap process will create 5 components: runtime_support (GID is
//    the prefix), memory (GID is the prefix with the memory bit set),
//    primary_namespace (GID is the prefix + 1), component_namespace (GID is the
//    prefix + 2) and symbol_namespace (GID is the prefix + 3). 

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/agas/namespace/primary.hpp>
#include <hpx/runtime/agas/database/backend/stdmap.hpp>
#include <hpx/runtime/agas/network/backend/tcpip.hpp>
#include <hpx/util/lightweight_test.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using boost::fusion::at_c;

using boost::asio::ip::address;

using hpx::naming::gid_type;
using hpx::naming::id_type;
using hpx::naming::locality;
using hpx::naming::strip_credit_from_gid;
using hpx::naming::get_prefix_from_gid;

using hpx::applier::get_applier;

using hpx::agas::primary_namespace;

using hpx::init;
using hpx::finalize;

using hpx::util::report_errors;

typedef primary_namespace<
    hpx::agas::tag::database::stdmap, hpx::agas::tag::network::tcpip
> primary_namespace_type;

typedef primary_namespace_type::endpoint_type endpoint_type;

typedef primary_namespace_type::binding_type binding_type;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    { 
       // Get this locality's prefix.
        id_type prefix = get_applier().get_runtime_support_gid();
    
        // Create the primary namespace.
        primary_namespace_type pri;
        pri.create(prefix);
    
        endpoint_type here(address::from_string("127.0.0.1"), 50000);
    
        // We need to register this locality + 3 GIDs. The return type is (lower
        // upper prefix is_first_bind).
        binding_type b = pri.bind(here, 3); 
        
        // This should be the first binding of this site.
        HPX_SANITY(at_c<3>(b));
            
        strip_credit_from_gid(at_c<0>(b));
        strip_credit_from_gid(at_c<1>(b));

        std::cout << "console prefix = " << get_prefix_from_gid(at_c<2>(b))
                << "\nlower id       = " << at_c<0>(b)
                << "\nupper id       = " << at_c<1>(b) << std::endl;
    }   
    
    { 
       // Get this locality's prefix.
        id_type prefix = get_applier().get_runtime_support_gid();
    
        // Create the primary namespace.
        primary_namespace_type pri;
        pri.create(prefix);
    
        endpoint_type there(address::from_string("127.0.0.1"), 50000);
        endpoint_type here(address::from_string("127.0.0.1"), 50001);

        // Register the console    
        binding_type there_bindings = pri.bind(there, 0); 

        // We need to register this locality + 3 GIDs. The return type is (lower
        // upper prefix is_first_bind).
        binding_type here_bindings = pri.bind(here, 3); 
        
        // This should be the first binding of the console and agas sites.
        HPX_SANITY(at_c<3>(there_bindings));
        HPX_SANITY(at_c<3>(here_bindings));
            
        strip_credit_from_gid(at_c<0>(here_bindings));
        strip_credit_from_gid(at_c<1>(here_bindings));

        std::cout << "console prefix = "
                  << get_prefix_from_gid(at_c<2>(there_bindings))
                << "\nagas prefix    = "
                  << get_prefix_from_gid(at_c<2>(here_bindings))
                << "\nlower id       = " << at_c<0>(here_bindings)
                << "\nupper id       = " << at_c<1>(here_bindings) << std::endl;
    }   
 
    // initiate shutdown of the runtime system
    finalize(1.0);
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(init(desc_commandline, argc, argv), 0,
      "HPX main exited with non-zero status");
    return report_errors();
}

