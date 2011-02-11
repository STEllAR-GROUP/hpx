#ifndef _TLF_COMPONENTS_020211
#define _TLF_COMPONENTS_020211

#include <hpx/runtime.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <boost/assert.hpp>

#include "stubs/tlf.hpp"

namespace hpx { namespace components { namespace tlf
{
   class tlf
   : public client_base<tlf, stubs::tlf>
   {
       typedef client_base<tlf, stubs::tlf> base_type;
       
   public:
       
       tlf()
       { }
       
       tlf(naming::id_type gid)
       :base_type(gid)
       { }
       
       ~tlf()
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
   };
}}}

#endif