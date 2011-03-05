#ifndef _NODE_COMPONENT_SERVER_020211
#define _NODE_COMPONENT_SERVER_020211
#include <iostream>

#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

namespace hpx { namespace components { namespace node { namespace server {
    
    class HPX_EXPORT node
    : public components::detail::managed_component_base<node>
    {
    public:
        enum actions
        {
          node_set_pos = 0,
          node_set_vel = 1,
          node_set_mass = 2,
          node_set_acc = 3,
          node_print = 4,
          node_get_pos = 5,
          node_get_type = 6,
          node_get_mass = 7
        };
        
        node();
        
        void set_mass(double mass_tmp);
        void set_pos(double px, double py, double pz); 
        double get_mass();
        std::vector<double> get_pos();
        int get_type();
        void set_vel(double vx, double vy, double vz);
        void set_acc(double ax, double ay, double az);
        void print();
        
        typedef hpx::actions::action3<node, node_set_pos, double, double, double, &node::set_pos> set_pos_action;
        typedef hpx::actions::action3<node, node_set_vel, double, double, double, &node::set_vel> set_vel_action;        
        typedef hpx::actions::action3<node, node_set_acc, double, double, double, &node::set_acc> set_acc_action;        
        typedef hpx::actions::action1<node, node_set_mass, double, &node::set_mass> set_mass_action;
        typedef hpx::actions::action0<node, node_print, &node::print> print_action;
        typedef hpx::actions::result_action0<node, std::vector<double>, node_get_pos, &node::get_pos> get_pos_action;
        typedef hpx::actions::result_action0<node, int, node_get_type, &node::get_type> get_type_action;
        typedef hpx::actions::result_action0<node, double, node_get_mass, &node::get_mass > get_mass_action;
        
    private:
        int node_type;
        double mass;
        double p[3];
        double v[3];
        double a[3];  
        
    };
    
}}}}
#endif