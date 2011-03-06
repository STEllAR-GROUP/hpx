#include <iostream>
#include <hpx/hpx_fwd.hpp>
#include <hpx/config.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include "node.hpp"
#include "../stubs/node.hpp"
#include "../node.hpp"

namespace hpx { namespace components { namespace node { namespace server {
    
        node::node()
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
        
        void node::set_mass(double mass_tmp)
        {
            mass = mass_tmp;
        }
        
        void node::set_pos(double px, double py, double pz)
        {
            p[0] = px;
            p[1] = py;
            p[2] = pz;
        }
        
        double node::get_mass()
        {
            return mass;
        }
        
        std::vector<double> node::get_pos()
        {
            std::vector<double> bPos;
            bPos.push_back(p[0]);
            bPos.push_back(p[1]);
            bPos.push_back(p[2]);    
            return bPos;
        }
        void node::set_type(int type_var)
        {
            node_type = type_var;
        }

        int node::get_type()
        {
            return node_type;
        }
        
        void node::set_vel(double vx, double vy, double vz)
        {
            v[0] = vx;
            v[1] = vy;
            v[2] = vz;
        }        
        
        void node::set_acc(double ax, double ay, double az)
        {
            a[0] = ax;
            a[1] = ay;
            a[2] = az;
        }        
                
        void node::print()
        {
            applier::applier& appl = applier::get_applier();
            
            if (node_type == 1)
            {
                std::cout << "BODY :: ";
                std::cout << appl.get_runtime_support_gid() 
                << " pos > " << p[0] << " " << p[1] << " " 
                << p[2] << " " << std::flush << std::endl;   
                std::cout << appl.get_runtime_support_gid() 
                << " vel > " << v[0] << " " << v[1] << " " 
                << v[2] << " " << std::flush << std::endl;     
            } else if (node_type == 0)
            {
                std::cout << "CELL :: ";
                std::cout << appl.get_runtime_support_gid() 
                << " pos > " << p[0] << " " << p[1] << " " 
                << p[2] << " " << std::flush << std::endl;   
            }
        }
        
        void node::new_node(double px, double py, double pz)
        {
            node_type = 0; // sets node type to indicate this node is a cell not a body
            mass = 0.0;
            p[0] = px;
            p[1] = py;
            p[2] = pz;
            for(int i = 0; i<8; ++i)
                child[i].state = -1;  // initialize all children to -1 implying that the children are empty
        }
        
        void node::insert_node(const hpx::naming::id_type& new_bod_gid, double sub_box_dim)
        {
            int i=0;
            std::vector<double> bpos = components::node::stubs::node::get_pos(new_bod_gid);

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
            if(child[i].state == -1)   // if child branch i is empty/NUll then set it to the new body being inserted
            {
                child[i].state = 1;
                child[i].gid = new_bod_gid;
            }
            else if(child[i].state == 0)   // if the child branch is an intermediate node
            {
                hpx::components::node::stubs::node::insert_node(child[i].gid, new_bod_gid, 0.5*sub_box_dim);
            }
            else if(child[i].state == 1) // if the child branch is a treeleaf
            {
                naming::id_type cur_bod_gid = child[i].gid;
                //create an intermediate node that will reside in the current branch
                child[i].state = 0;
                const double new_sub_box_dim = 0.5 * sub_box_dim;
                const double pos_buf[] = {p[0] - new_sub_box_dim + temp[0], p[1] - new_sub_box_dim + temp[1], p[2] - new_sub_box_dim + temp[2] };
                hpx::components::node::node temp_node;
                hpx::naming::id_type prefix;
                hpx::applier::applier& appl = hpx::applier::get_applier();
                prefix = appl.get_runtime_support_gid();
                temp_node.create(prefix);    
                temp_node.new_node(pos_buf[0], pos_buf[1], pos_buf[2]);
                child[i].gid = temp_node.get_gid(); 
                
                temp_node.insert_node(cur_bod_gid, new_sub_box_dim);
                temp_node.insert_node(new_bod_gid, new_sub_box_dim); 
            }
        }

        void node::calc_cm(naming::id_type current_node)
        {
            double m, px = 0.0, py = 0.0, pz = 0.0;
            int var = 0;
            if (node_type == 0)
                mass = 0.0;
            for (int i=0; i < 8; ++i)
            {
                if (child[i].state != -1)
                {
                    if (child[i].state == 0 )
                    {
                        std::cout << "I am a CELL" << std::endl;
                        components::node::stubs::node::calc_cm(child[i].gid, current_node);
                    }
                    else if (child[i].state == 1)
                    {
                        std::cout << "I am a BODY" << std::endl;
                    }

                }
            }
        }

}}}}