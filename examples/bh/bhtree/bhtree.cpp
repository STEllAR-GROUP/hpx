#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/bhtree.hpp"


///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::managed_component<
    hpx::components::server::IntrTreeNode
> IntrTreeNode_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(IntrTreeNode_type, IntrTreeNode);

HPX_REGISTER_ACTION_EX(
   IntrTreeNode_type::wrapped_type::newNode_action, 
   IntrTreeNode_newNode_action);

HPX_REGISTER_ACTION_EX(
    IntrTreeNode_type::wrapped_type::print_action,
    IntrTreeNode_print_action);
    
HPX_DEFINE_GET_COMPONENT_TYPE(IntrTreeNode_type::wrapped_type);

