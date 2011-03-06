#include <hpx/hpx.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/node.hpp"

HPX_REGISTER_COMPONENT_MODULE();
typedef hpx::components::managed_component<hpx::components::node::server::node> node_type;
HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(node_type, node);
HPX_DEFINE_GET_COMPONENT_TYPE(node_type::wrapped_type);
HPX_REGISTER_ACTION_EX(node_type::wrapped_type::set_mass_action, node_set_mass_action);
HPX_REGISTER_ACTION_EX(node_type::wrapped_type::set_type_action, node_set_type_action);
HPX_REGISTER_ACTION_EX(node_type::wrapped_type::set_pos_action, node_set_pos_action);
HPX_REGISTER_ACTION_EX(node_type::wrapped_type::set_vel_action, node_set_vel_action);
HPX_REGISTER_ACTION_EX(node_type::wrapped_type::set_acc_action, node_set_acc_action);
HPX_REGISTER_ACTION_EX(node_type::wrapped_type::print_action, node_print_action);
HPX_REGISTER_ACTION_EX(node_type::wrapped_type::get_pos_action, node_get_pos_action);
HPX_REGISTER_ACTION_EX(node_type::wrapped_type::get_type_action, node_get_type_action);
HPX_REGISTER_ACTION_EX(node_type::wrapped_type::get_mass_action, node_get_mass_action);
HPX_REGISTER_ACTION_EX(node_type::wrapped_type::new_node_action, node_new_node_action);


HPX_REGISTER_ACTION_EX(hpx::lcos::base_lco_with_value<double>::set_result_action, set_result_action_double);
HPX_REGISTER_ACTION_EX(hpx::lcos::base_lco_with_value<int>::set_result_action, set_result_action_int);
HPX_REGISTER_ACTION_EX(hpx::lcos::base_lco_with_value<std::vector<double> >::set_result_action, set_result_action_vector_double);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco_with_value<std::vector<double> >);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco_with_value<double>);
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::lcos::base_lco_with_value<int>);