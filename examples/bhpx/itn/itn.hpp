#ifndef _ITN_COMPONENTS_020411
#define _ITN_COMPONENTS_020411

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <boost/assert.hpp>

#include "stubs/itn.hpp"

namespace hpx { namespace components { namespace itn
{
   class itn
   : public client_base<itn, stubs::itn>
   {
       typedef client_base<itn, stubs::itn> base_type;
       
   public:
       itn()
       { }
       
       itn(naming::id_type gid)
       : base_type(gid)
       { }
       
       void set_mass(double mass_buf)
       {
           BOOST_ASSERT(gid_);
           this->base_type::set_mass(gid_, mass_buf);
       }
       
       void set_pos(double px, double py, double pz)
       {
           BOOST_ASSERT(gid_);
           this->base_type::set_pos(gid_, px, py, pz);
       }
       
       void new_node(double px, double py, double pz)
       {
           BOOST_ASSERT(gid_);
           this->base_type::new_node(gid_, px, py, pz);
       }
       
       void insert_body(naming::id_type const & new_bod_gid, double sub_box_dim)
       {
           BOOST_ASSERT(gid_);
           this->base_type::insert_body(gid_, new_bod_gid, sub_box_dim);
       }
        
       void calc_cm(naming::id_type current_node)
       {
           BOOST_ASSERT(gid_);
           this->base_type::calc_cm(gid_, current_node);
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
       
       double get_mass()
       {
           BOOST_ASSERT(gid_);
           this->base_type::get_mass(gid_);
       }
       
       lcos::future_value<double> get_mass_async()
       {
           return this->base_type::get_mass_async(gid_);
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
   };
}}}


#endif