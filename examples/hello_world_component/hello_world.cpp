//[hello_world_cpp_getting_started
#include "hello_world.hpp"
#include <hpx/include/iostreams.hpp>

namespace examples { namespace server
{

void hello_world::invoke()
{
    hpx::cout << "Hello World!\n" << hpx::flush;
}

}}

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::managed_component<
    examples::server::hello_world
> hello_world_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(hello_world_type, hello_world);

HPX_DEFINE_GET_COMPONENT_TYPE(hello_world_type::wrapped_type);

HPX_REGISTER_ACTION_EX(
    examples::server::hello_world::invoke_action, hello_world_invoke_action);
//]

