//  Copyright (c) 2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>
    
using hpx::naming::locality;
using hpx::naming::address;
using hpx::naming::resolver_client;
using hpx::naming::gid_type;
using hpx::naming::server::get_command_name;
using hpx::naming::server::command_lastcommand;
using hpx::naming::get_prefix_from_gid;
using hpx::naming::strip_credit_from_gid;

using hpx::detail::agas_server_helper;

using hpx::util::get_random_port;
using hpx::util::io_service_pool;

using hpx::util::report_errors;

// no-op
int hpx_main(boost::program_options::variables_map &vm) { return 0; }

int main()
{
    // This is our locality.
    locality here("127.0.0.1", 50000);

    // Start the AGAS server in the background.
    agas_server_helper agas_server("127.0.0.1", 50000);

    // Create a client and connect it to the server.
    io_service_pool io_service_pool; 
    resolver_client resolver(io_service_pool, here); 
    
    // Retrieve the id prefix of this site.
    gid_type prefix;
    resolver.get_prefix(here, prefix);
    
    std::cout << "console prefix = " << get_prefix_from_gid(prefix) << "\n";

    gid_type lower, upper;
    HPX_TEST(!resolver.get_id_range(here, 3, lower, upper));
    
    // Strip the credits from the range.
    strip_credit_from_gid(lower);
    strip_credit_from_gid(upper);
    
    std::cout << "lower id       = " << lower << "\n"
              << "upper id       = " << upper << std::endl;

    return report_errors();
}

