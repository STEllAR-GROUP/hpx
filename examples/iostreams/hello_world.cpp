///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach 
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include <boost/ref.hpp>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/components/server/manage_component.hpp>
#include <hpx/components/iostreams/lazy_ostream.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::init;
using hpx::finalize;

using hpx::naming::id_type;

using hpx::components::server::create_one;
using hpx::components::managed_component;

using hpx::endl;
using hpx::iostreams::lazy_ostream;
using hpx::iostreams::server::output_stream;

typedef managed_component<output_stream> ostream_type;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map&)
{
    {
        lazy_ostream hpx_cout
            (id_type(create_one<ostream_type>(boost::ref(std::cout))
                   , id_type::managed));

        hpx_cout << "hello world!" << endl;
    }

    // Initiate shutdown of the runtime system.
    finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}

