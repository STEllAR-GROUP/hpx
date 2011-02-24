#ifndef _TLF_COMPONENT_SERVER_020211
#define _TLF_COMPONENT_SERVER_020211
#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

namespace hpx { namespace components { namespace tlf { namespace server {
    
    class HPX_ALWAYS_EXPORT tlf
    : public components::detail::managed_component_base<tlf>
    {
    public:
        enum actions
        {
          tlf_set_pos = 0,
          tlf_set_vel = 1,
          tlf_set_mass = 2,
          tlf_set_acc = 3,
          tlf_print = 4,
          tlf_get_pos = 5,
          tlf_get_type = 6
        };
        
        tlf()
        {
            node_type = 1;
            mass = 0.0;
            p[0] = 0.0;
            p[1] = 0.0;
            p[2] = 0.0;
            v[0] = 0.0;
            v[1] = 0.0;
            v[2] = 0.0;
            a[0] = 0.0;
            a[1] = 0.0;
            a[2] = 0.0;
        }
        
        void set_mass(double mass_tmp)
        {
            mass = mass_tmp;
        }
        
        void set_pos(double px, double py, double pz)
        {
            p[0] = px;
            p[1] = py;
            p[2] = pz;
        }
        
        std::vector<double> get_pos()
        {
            std::vector<double> bPos;
            bPos.push_back(p[0]);
            bPos.push_back(p[1]);
            bPos.push_back(p[2]);    
            return bPos;
        }
        
        int get_type()
        {
            return node_type;
        }
        
        void set_vel(double vx, double vy, double vz)
        {
            v[0] = vx;
            v[1] = vy;
            v[2] = vz;
        }        
        
        void set_acc(double ax, double ay, double az)
        {
            a[0] = ax;
            a[1] = ay;
            a[2] = az;
        }        
                
        void print()
        {
            applier::applier& appl = applier::get_applier();
            std::cout << appl.get_runtime_support_gid() 
            << " pos > " << p[0] << " " << p[1] << " " 
            << p[2] << " " << std::flush << std::endl;   
            std::cout << appl.get_runtime_support_gid() 
            << " vel > " << v[0] << " " << v[1] << " " 
            << v[2] << " " << std::flush << std::endl;     
        }
        
        typedef hpx::actions::action3<tlf, tlf_set_pos, double, double, double, &tlf::set_pos> set_pos_action;
        typedef hpx::actions::action3<tlf, tlf_set_vel, double, double, double, &tlf::set_vel> set_vel_action;        
        typedef hpx::actions::action3<tlf, tlf_set_acc, double, double, double, &tlf::set_acc> set_acc_action;        
        typedef hpx::actions::action1<tlf, tlf_set_mass, double, &tlf::set_mass> set_mass_action;
        typedef hpx::actions::action0<tlf, tlf_print, &tlf::print> print_action;
        typedef hpx::actions::result_action0<tlf, std::vector<double>, tlf_get_pos, &tlf::get_pos> get_pos_action;
        typedef hpx::actions::result_action0<tlf, int, tlf_get_type, &tlf::get_type> get_type_action;
        
    private:
        int node_type;
        double mass;
        double p[3];
        double v[3];
        double a[3];        
    };
}}}}
#endif
