#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/hplmatrex3.hpp"

///////////////////////////////////////////////////////////////
HPX_REGISTER_COMPONENT_MODULE();

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
	hpx::components::simple_component<hpx::components::server::HPLMatreX3>,
	HPLMatreX3);

//Register the actions
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatreX3::construct_action,HPLconstruct_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatreX3::destruct_action,HPLdestruct_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatreX3::assign_action,HPLassign_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatreX3::set_action,HPLset_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatreX3::get_action,HPLget_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatreX3::solve_action,HPLsolve_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatreX3::swap_action,HPLswap_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatreX3::ghub_action,HPLghub_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatreX3::bsubst_action,HPLbsubst_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatreX3::check_action,HPLcheck_action);

HPX_DEFINE_GET_COMPONENT_TYPE(hpx::components::server::HPLMatreX3);
