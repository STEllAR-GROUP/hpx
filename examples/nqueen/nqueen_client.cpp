///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2011 Hartmut Kaiser, Richard D Guidry Jr.
//  Copyright (c) 2011 Bryce Lelbach 
//  Copyright (c) 2011 Vinay C Amatya
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>

#include <nqueen/board.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;
using hpx::find_here;

using hpx::nqueen::board;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        const std::size_t size = vm["board-size"].as<std::size_t>();

        board b;

        b.create_one(find_here(), size);

        b.solve(b.access(), size, 0);

        b.print();
    }

    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "board-size"
        , value<std::size_t>()->default_value(8) 
        , "size of the board")
        ;

    // Initialize and run HPX.
    return init(desc_commandline, argc, argv);
}

