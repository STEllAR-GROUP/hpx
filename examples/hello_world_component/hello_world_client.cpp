//[hello_world_client_getting_started
#include "hello_world.hpp"
#include <hpx/hpx_init.hpp>

int hpx_main(boost::program_options::variables_map&)
{
    {
        examples::hello_world client;

        // Create a single instance of the component on this locality.
        client.create(hpx::find_here());

        // Invoke the components action, which will print "Hello World!".
        client.invoke();
    }

    return hpx::finalize(); // Initiate shutdown of the runtime system.
}

int main(int argc, char* argv[])
{
    // Configure application-specific options.
    boost::program_options::options_description desc_commandline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    return hpx::init(desc_commandline, argc, argv); // Initialize and run HPX.
}
//]

