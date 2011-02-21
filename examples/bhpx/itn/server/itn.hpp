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

#include "../itn.hpp"
#include "../stubs/itn.hpp"
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
        
        itn()
        {
            node_type = 0;
            mass = 0.0;
            p[0] = 0.0;
            p[1] = 0.0;
            p[2] = 0.0;            
        }
        
        void new_node(double px, double py, double pz)
        {
            node_type = 0;
            mass = 0.0;
            p[0] = px;
            p[1] = py;
            p[2] = pz;
            for(int i = 0; i<8; ++i)
                child[i].leaf = -1;  // initialize all children to -1 implying that the children are empty
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
        
        double get_mass()
        {
            return mass;
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
        
        
        void print()
        {
            applier::applier& appl = applier::get_applier();
            std::cout << appl.get_runtime_support_gid() 
            << " pos > " << p[0] << " " << p[1] << " " 
            << p[2] << " " << std::flush << std::endl;    
        }
        
        void insert_body(naming::id_type const & new_bod_gid, double sub_box_dim)
        {
            int i=0;
            std::vector<double> bpos = components::tlf::stubs::tlf::get_pos(new_bod_gid);
            
            double temp[3];
            temp[0] = 0.0;
            temp[1] = 0.0;
            temp[2] = 0.0;
            
            if(p[0] < bpos[0])
            {
                i = 1;
                temp[0] = sub_box_dim;
            }
            if(p[1] < bpos[1])
            {
                i += 2;
                temp[2] = sub_box_dim;
            }
            if(p[2] < bpos[2])
            {
                i += 4;
                temp[2] = sub_box_dim;
            }
            
            if(child[i].leaf == -1)   // if child branch i is empty/NUll then set it to the new body being inserted
            {
                child[i].leaf = 1;
                child[i].gid = new_bod_gid;
            }
            
            else if(child[i].leaf == 0)   // if the child branch is an intermediate node
            {
                // Not correct the key issue here would be to get the ITN component from the child[i].gid and then 
                // call the insert_body function on the retrieved component 
                // TODO: need to create child[i].gid
                
                
               hpx::components::itn::stubs::itn::insert_body(child[i].gid, new_bod_gid, 0.5* sub_box_dim);
                //insert_body(child[i].gid, 0.5 * sub_box_dim);
            }
            
            else if(child[i].leaf == 1) //if the child branch node is a treeleaf
            {

                // TODO: need to create child[i].gid
                
                naming::id_type cur_bod_gid = child[i].gid;
                
                //create an intermediate node that will reside in the current branch
                child[i].leaf = 0;
                const double new_sub_box_dim = 0.5 * sub_box_dim;
                const double pos_buf[] = {p[0] - new_sub_box_dim + temp[0], p[1] - new_sub_box_dim + temp[1], p[2] - new_sub_box_dim + temp[2] };
                hpx::components::itn::itn temp_itn;
                
                
                temp_itn.create(child[i].gid);
                temp_itn.new_node(pos_buf[0], pos_buf[1], pos_buf[2]);
                child[i].gid = temp_itn.get_gid(); 
                
                
                temp_itn.insert_body(cur_bod_gid, new_sub_box_dim);
                temp_itn.insert_body(new_bod_gid, new_sub_box_dim);
                
                
            } 
            
            
        }
             
        typedef hpx::actions::action3<itn, itn_set_pos, double, double, double, &itn::set_pos> set_pos_action;
        typedef hpx::actions::action1<itn, itn_set_mass, double, &itn::set_mass> set_mass_action;
        typedef hpx::actions::result_action0<itn, double, itn_get_mass, &itn::get_mass > get_mass_action;
        typedef hpx::actions::result_action0<itn, std::vector<double>, itn_get_pos, &itn::get_pos> get_pos_action;
        typedef hpx::actions::action0<itn, itn_print, &itn::print> print_action;
        typedef hpx::actions::action3<itn, itn_new_node, double, double, double, &itn::new_node> new_node_action;
        typedef hpx::actions::result_action0<itn, int, itn_get_type, &itn::get_type> get_type_action;
        typedef hpx::actions::action2<itn, itn_insert_body, naming::id_type, double, &itn::insert_body > insert_body_action;
        
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