#include <hpx/runtime/components/component_factory.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>

#include "server/hplmatrex.hpp"

///////////////////////////////////////////////////////////////
HPX_REGISTER_COMPONENT_MODULE();

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(
	hpx::components::simple_component<hpx::components::server::HPLMatrex>,
	HPLMatrex);

//Register the actions
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatrex::construct_action,HPLconstruct_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatrex::destruct_action,HPLdestruct_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatrex::mkvec_action,HPLmkvek_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatrex::allocate_action,HPLallocate_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatrex::assign_action,HPLassign_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatrex::set_action,HPLset_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatrex::get_action,HPLget_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatrex::solve_action,HPLsolve_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatrex::gauss_action,HPLgauss_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatrex::gline_action,HPLgline_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatrex::bsubst_action,HPLbsubst_action);
HPX_REGISTER_ACTION_EX(
	hpx::components::server::HPLMatrex::check_action,HPLcheck_action);

HPX_DEFINE_GET_COMPONENT_TYPE(hpx::components::server::HPLMatrex);
