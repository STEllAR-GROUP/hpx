#ifndef _ITN_COMPONENT_STUB_020411
#define _ITN_COMPONENT_STUB_020411

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include "../server/itn.hpp"

namespace hpx { namespace components { namespace itn { namespace stubs {
    
    struct itn: components::stubs::stub_base<hpx::components::itn::server::itn>
    {
        static void set_mass(naming::id_type const& gid, double mass_tmp)
        {
            applier::apply<hpx::components::itn::server::itn::set_mass_action>(gid, mass_tmp);
        }
        
        static void set_pos(naming::id_type const& gid, double px, double py, double pz)
        {
            applier::apply<hpx::components::itn::server::itn::set_pos_action>(gid, px, py, pz);
        }
        
        static void new_node(naming::id_type const & gid, double px, double py, double pz)
        {
            applier::apply<hpx::components::itn::server::itn::new_node_action>(gid, px, py, pz);
        }
        
        static void print(naming::id_type gid)
        {
            applier::apply<hpx::components::itn::server::itn::print_action>(gid);
        }
        
        static lcos::future_value<std::vector<double> > get_pos_async(naming::id_type const& gid)
        {
            typedef components::itn::server::itn::get_pos_action action_type;
            return lcos::eager_future<action_type>(gid);
        }
        
        static std::vector<double> get_pos(naming::id_type const& gid)
        {
            return get_pos_async(gid).get();
        }
        
        static lcos::future_value<double> get_mass_async(naming::id_type const& gid)
        {
            typedef components::itn::server::itn::get_mass_action action_type;
            return lcos::eager_future<action_type>(gid);
        }
        
        static double get_mass(naming::id_type const& gid)
        {
            return get_mass_async(gid).get();
        }
        
        static lcos::future_value<int> get_type_async(naming::id_type const& gid)
        {
            typedef components::itn::server::itn::get_type_action action_type;
            return lcos::eager_future<action_type>(gid);
        }
        
        static int get_type(naming::id_type const& gid)
        {
            return get_type_async(gid).get();
        }
        
        static void insert_body(naming::id_type const & gid,  naming::id_type const & new_bod_gid, double sub_box_dim)
        {
            applier::apply<hpx::components::itn::server::itn::insert_body_action>(gid, new_bod_gid, sub_box_dim);
        }
    };
}}}}

#endif