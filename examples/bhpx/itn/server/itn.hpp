#ifndef _ITN_SERVER_HPP_020411
#define _ITN_SERVER_HPP_020411

#include <iostream>
#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include "../../tlf/tlf.hpp"

// #include "../itn.hpp"
// #include "../stubs/itn.hpp"
// #include "../stubs/itn.hpp"

namespace hpx { namespace components { namespace itn { namespace server
{
    
    class itn
    : public components::detail::managed_component_base<itn>
    {
    public:
        
        struct child_type 
        {
          int leaf;      // leaf = 0 - Intermediate Node, leaf = -1 - NULL, leaf = 1 - TreeLeaf type
          naming::id_type gid;
        };
        
        enum actions
        {
            itn_new_node = 0,
            itn_set_mass = 1,
            itn_set_pos = 2,
            itn_get_mass = 3,
            itn_get_pos = 4,
            itn_print = 5,
            itn_get_type = 6,
            itn_insert_body = 7
        };
        
        itn();
        void new_node(double px, double py, double pz);
        void set_mass(double mass_tmp);
        void set_pos(double px, double py, double pz);
        double get_mass();
        std::vector<double> get_pos();
        int get_type();
        void print();
        void insert_body(naming::id_type const & new_bod_gid, double sub_box_dim);
             
        typedef hpx::actions::action3<itn, itn_set_pos, double, double, double, &itn::set_pos> set_pos_action;
        typedef hpx::actions::action1<itn, itn_set_mass, double, &itn::set_mass> set_mass_action;
        typedef hpx::actions::result_action0<itn, double, itn_get_mass, &itn::get_mass > get_mass_action;
        typedef hpx::actions::result_action0<itn, std::vector<double>, itn_get_pos, &itn::get_pos> get_pos_action;
        typedef hpx::actions::action0<itn, itn_print, &itn::print> print_action;
        typedef hpx::actions::action3<itn, itn_new_node, double, double, double, &itn::new_node> new_node_action;
        typedef hpx::actions::result_action0<itn, int, itn_get_type, &itn::get_type> get_type_action;
        typedef hpx::actions::action2<itn, itn_insert_body, naming::id_type const &, double, &itn::insert_body > insert_body_action;
        
    private:
        int node_type;
        double mass;
        double p[3];
        child_type child[8];
        //naming::id_type child[8];               // reference to 8 children nodes
        naming::id_type parent;     
    };
}}}}

#endif