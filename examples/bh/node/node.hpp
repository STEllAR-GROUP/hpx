#ifndef _node_COMPONENTS_020211
#define _node_COMPONENTS_020211

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <boost/assert.hpp>

#include "stubs/node.hpp"

namespace hpx { namespace components { namespace node
{
   class node
   : public client_base<node, stubs::node>
   {
       typedef client_base<node, stubs::node> base_type;
       
   public:
       node()
       { }
       
       node(naming::id_type gid)
       : base_type(gid)
       { }
       
       void set_mass(double mass_buf)
       {
           BOOST_ASSERT(gid_);
           this->base_type::set_mass(gid_, mass_buf);
       }
       
       void set_type(int type_var)
       {
           BOOST_ASSERT(gid_);
           this->base_type::set_type(gid_, type_var);
       }
       
       void set_pos(double px, double py, double pz)
       {
           BOOST_ASSERT(gid_);
           this->base_type::set_pos(gid_, px, py, pz);           
       }
       
       void set_vel(double vx, double vy, double vz)
       {
           BOOST_ASSERT(gid_);
           this->base_type::set_vel(gid_, vx, vy, vz);
       }
       
       void set_acc(double ax, double ay, double az)
       {
           BOOST_ASSERT(gid_);
           this->base_type::set_acc(gid_, ax, ay, az);
       }
       
       void print()
       {
           BOOST_ASSERT(gid_);
           this->base_type::print(gid_);
       }
       
       std::vector<double> get_pos()
       {
           BOOST_ASSERT(gid_);
           this->base_type::get_pos(gid_);
       }
      
       lcos::future_value<std::vector<double> > get_pos_async()
       {
           return this->base_type::get_pos_async(gid_);
       }
       
       int get_type()
       {
           BOOST_ASSERT(gid_);
           this->base_type::get_type(gid_);
       }
       
       lcos::future_value<int> get_type_async()
       {
           BOOST_ASSERT(gid_);
           this->base_type::get_type_async(gid_);
       }
       
       double get_mass()
       {
           BOOST_ASSERT(gid_);
           this->base_type::get_mass(gid_);
       }
       
       lcos::future_value<double> get_mass_async()
       {
           return this->base_type::get_mass_async(gid_);
       }
           
       void new_node(double px, double py, double pz)
       {
           BOOST_ASSERT(gid_);
           this->base_type::new_node(gid_, px, py, pz);
       }
       
       void insert_node(naming::id_type const & new_bod_gid, double sub_box_dim)
       {
           BOOST_ASSERT(gid_);
           this->base_type::insert_node(gid_, new_bod_gid, sub_box_dim);
       }
   };
}}}

#endif