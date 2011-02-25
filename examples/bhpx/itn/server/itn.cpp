
#include <iostream>
#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include "itn.hpp"  // serverhpp
#include "../itn.hpp"
#include "../stubs/itn.hpp"

#include "../../tlf/tlf.hpp"
#include "../../tlf/stubs/tlf.hpp"

namespace hpx { namespace components { namespace itn { namespace server
{ 
        itn::itn()
        {
            node_type = 0;
            mass = 0.0;
            p[0] = 0.0;
            p[1] = 0.0;
            p[2] = 0.0;            
        }
        
        void itn::new_node(double px, double py, double pz)
        {
            node_type = 0;
            mass = 0.0;
            p[0] = px;
            p[1] = py;
            p[2] = pz;
            for(int i = 0; i<8; ++i)
                child[i].leaf = -1;  // initialize all children to -1 implying that the children are empty
        }
        
        void itn::set_mass(double mass_tmp)
        {
            mass = mass_tmp;
        }
        
        void itn::set_pos(double px, double py, double pz)
        {
            p[0] = px;
            p[1] = py;
            p[2] = pz;
        }        
        
        double itn::get_mass()
        {
            return mass;
        }
        
        std::vector<double> itn::get_pos()
        {
            std::vector<double> bPos;
            bPos.push_back(p[0]);
            bPos.push_back(p[1]);
            bPos.push_back(p[2]);    
            return bPos;
        }
        
        int itn::get_type()
        {
            return node_type;
        }
        
        
        void itn::print()
        {
            applier::applier& appl = applier::get_applier();
            std::cout << appl.get_runtime_support_gid() 
            << " pos > " << p[0] << " " << p[1] << " " 
            << p[2] << " " << std::flush << std::endl;    
        }
        
        void itn::insert_body(naming::id_type const & new_bod_gid, double sub_box_dim)
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
        

        
        void itn::calc_cm(naming::id_type current_node)
        {
            double m, px = 0.0, py = 0.0, pz = 0.0;
            int var = 0;
            mass = 0.0;
            
            for (int i=0; i < 8; ++i)
            {
                if (child[i].leaf == 0)
                    hpx::components::itn::itn temp_branch;
                else if (child[i].leaf == 1)
                    hpx::components::tlf::tlf temp_branch;
                
                if (child[i].leaf == 0) // contains an intermediate tree node (itn)
                {
                    components::itn::stubs::itn::calc_cm(child[i].gid, current_node);
                }
                
                if (child[i].leaf == 0)
                    m = components::itn::stubs::itn::get_mass(child[i].gid);
                else if (child[i].leaf == 1)
                    m = components::tlf::stubs::tlf::get_mass(child[i].gid);
                
               mass += m;
               
               std::vector<double> bpos;
               if (child[i].leaf == 0)
                   bpos = components::itn::stubs::itn::get_pos(child[i].gid);
               else if (child[i].leaf == 1)
                   bpos = components::tlf::stubs::tlf::get_pos(child[i].gid);              
               
               
               px += bpos[0] * m;
               py += bpos[1] * m;
               pz += bpos[2] * m;
               
            }
            
            m = 1.0 / mass;
            
            p[0] = px * m;
            p[1] = py * m;
            p[2] = pz * m;
        }
    
}}}}