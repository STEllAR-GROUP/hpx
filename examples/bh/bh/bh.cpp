#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/bh.hpp"


///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
    hpx::components::managed_component<hpx::components::server::IntrTreeNode>, 
    IntrTreeNode);

HPX_REGISTER_ACTION_EX(
    hpx::components::server::IntrTreeNode::newNode_action,
    IntrTreeNode_newNode_action);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::components::server::IntrTreeNode);



