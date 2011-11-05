
#include <hpx/hpx_fwd.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include "managed_test_component.hpp"

using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::init;
using hpx::finalize;

int hpx_main(variables_map & vm)
{
    cout << "start of hpx_main\n" << flush;
    {
        test_component t1;
        simple_test_component t2;
    }
    cout << "end of hpx_main\n" << flush;
    finalize();
    return 0;
}

int main(int argc, char ** argv)
{
    options_description
        cmdline("usage: " HPX_APPLICATION_STRING " [options]");
    
    return init(cmdline, argc, argv);
}
