#ifndef _TLF_COMPONENT_STUB_020211
#define _TLF_COMPONENT_STUB_020211

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>

#include "../server/tlf.hpp"


namespace hpx {
namespace components {
namespace tlf {
namespace stubs
{
struct tlf : components::stubs::stub_base<hpx::components::tlf::server::tlf>
{
    static void set_mass(naming::id_type const& gid, double mass_tmp)
    {
        applier::apply<hpx::components::tlf::server::tlf::set_mass_action>(gid, mass_tmp);
    }

    static void set_pos(naming::id_type gid, double px, double py, double pz)
    {
        applier::apply<hpx::components::tlf::server::tlf::set_pos_action>(gid, px, py, pz);
    }

    static void set_vel(naming::id_type gid, double vx, double vy, double vz)
    {
        applier::apply<hpx::components::tlf::server::tlf::set_vel_action>(gid, vx, vy, vz);
    }

    static void set_acc(naming::id_type gid, double ax, double ay, double az)
    {
        applier::apply<hpx::components::tlf::server::tlf::set_acc_action>(gid, ax, ay, az);
    }

    static void print(naming::id_type gid)
    {
        applier::apply<hpx::components::tlf::server::tlf::print_action>(gid);
    }

    static lcos::future_value<std::vector<double> > get_pos_async(naming::id_type const& gid)
    {
        typedef components::tlf::server::tlf::get_pos_action action_type;
        return lcos::eager_future<action_type>(gid);
        //applier::apply<hpx::components::tlf::server::tlf::get_pos_action>(gid);
    }

    static std::vector<double> get_pos(naming::id_type const& gid)
    {
        return get_pos_async(gid).get();
    }
    
    static lcos::future_value<int> get_type_async(naming::id_type const& gid)
    {
        typedef components::tlf::server::tlf::get_type_action action_type;
        return lcos::eager_future<action_type>(gid);
    }
    
    static int get_type(naming::id_type const& gid)
    {
        return get_type_async(gid).get();
    }
    
    static lcos::future_value<double> get_mass_async(naming::id_type const& gid)
    {
       typedef components::tlf::server::tlf::get_mass_action action_type;
       return lcos::eager_future<action_type>(gid);
    }
        
    static double get_mass(naming::id_type const& gid)
    {
       return get_mass_async(gid).get();
    }    
};
}
}
}
}
#endif
