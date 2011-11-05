
#include "managed_test_component.hpp"

HPX_REGISTER_COMPONENT_MODULE()

typedef hpx::components::managed_component<server::test_component> server_test_component;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(server_test_component, server_test_component)
HPX_DEFINE_GET_COMPONENT_TYPE(server::test_component)

typedef hpx::components::simple_component<server::simple_test_component> server_simple_test_component;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(server_simple_test_component, server_simple_test_component)
HPX_DEFINE_GET_COMPONENT_TYPE(server::simple_test_component)

