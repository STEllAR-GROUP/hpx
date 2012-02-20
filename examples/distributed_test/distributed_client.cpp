// test


#include <hpx/hpx_init.hpp>
#include <boost/assign/std.hpp>
#include "distributed_test/distribution.hpp"

using boost::program_options::variables_map;
using boost::program_options::options_description;

///////////////////////////////////////////////////////////////////////////////

int hpx_main(variables_map& vm)
{
    std::size_t num_localities = 2;
    std::size_t my_cardinality = std::size_t(-1);
    std::size_t init_length = 10, init_value = 5; 
    //--------------------------
    char const* distrib_symbolic_name = "distributed_symbolic_name";
   
    distributed::distribution distrib;
    distrib.create(distrib_symbolic_name, num_localities, my_cardinality,
        init_length, init_value);

    //------------------------
    hpx::finalize();
    return 0;
}
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    //aplication specific options
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    //Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
