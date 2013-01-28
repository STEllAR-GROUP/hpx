

#include <hpx/hpx_init.hpp>
#include <boost/assign/std.hpp>

#include <iostream>
#include <fstream>

#include "tictactoe_component/server/tictactoe.hpp"
#include "tictactoe_component/dist_factory.hpp"

using boost::program_options::variables_map;
using boost::program_options::options_description;

////////////////////////////////////////////////////////////////////////////////

int hpx_main(boost::program_options::variables_map& vm)
{
    {
        char winner;
        game::dist_factory df;
        std::cout << "begin!" << std::endl;
        winner = df.create(); 
        std::cout << "Winner of game is:" << winner << std::endl;
    }
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using boost::program_options::value;

    //Application specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");
    
    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv);
}
