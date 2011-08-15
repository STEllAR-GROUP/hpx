//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Matthew Anderson
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/lcos/future_wait.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/export.hpp>

#include <boost/foreach.hpp>

#include <math.h>

#include "stencil.hpp"
#include "logging.hpp"
#include "stencil_data.hpp"
#include "stencil_functions.hpp"
#include "stencil_data_locking.hpp"
#include "../mesh/unigrid_mesh.hpp"

#if defined(RNPL_FOUND)
#include <sdf.h>
#endif

// floatcmp_le {{{
HPX_COMPONENT_EXPORT bool floatcmp_le(double const& x1, double const& x2) {
  // compare two floating point numbers
  static double const epsilon = 1.e-8;

  if ( x1 < x2 ) return true;

  if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
    // the numbers are close enough for coordinate comparison
    return true;
  } else {
    return false;
  }
}
// }}}
 
// floatcmp {{{
HPX_COMPONENT_EXPORT int floatcmp(double const& x1, double const& x2) {
  // compare two floating point numbers
  static double const epsilon = 1.e-8;
  if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
    // the numbers are close enough for coordinate comparison
    return true;
  } else {
    return false;
  }
}
// }}}

// intersection {{{
HPX_COMPONENT_EXPORT bool intersection(double xmin,double xmax,
                  double ymin,double ymax,
                  double zmin,double zmax,
                  double xmin2,double xmax2,
                  double ymin2,double ymax2,
                  double zmin2,double zmax2)
{
  double pa[3],ea[3];
  static double const half = 0.5;
  pa[0] = half*(xmax + xmin);
  pa[1] = half*(ymax + ymin);
  pa[2] = half*(zmax + zmin);

  ea[0] = xmax - pa[0];
  ea[1] = ymax - pa[1];
  ea[2] = zmax - pa[2];

  double pb[3],eb[3];
  pb[0] = half*(xmax2 + xmin2);
  pb[1] = half*(ymax2 + ymin2);
  pb[2] = half*(zmax2 + zmin2);

  eb[0] = xmax2 - pb[0];
  eb[1] = ymax2 - pb[1];
  eb[2] = zmax2 - pb[2];

  double T[3];
  T[0] = pb[0] - pa[0];
  T[1] = pb[1] - pa[1];
  T[2] = pb[2] - pa[2];

  if ( floatcmp_le(fabs(T[0]),ea[0] + eb[0]) &&
       floatcmp_le(fabs(T[1]),ea[1] + eb[1]) &&
       floatcmp_le(fabs(T[2]),ea[2] + eb[2]) ) {
    return true;
  } else {
    return false;
  }

}
// }}}

// max {{{
HPX_COMPONENT_EXPORT double (max)(double x1, double x2) {
  if ( x1 > x2 ) return x1;
  else return x2;
}
// }}}

// min {{{
HPX_COMPONENT_EXPORT double (min)(double x1, double x2) {
  if ( x1 < x2 ) return x1;
  else return x2;
}
// }}}


///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    ///////////////////////////////////////////////////////////////////////////
    stencil::stencil()
      : numsteps_(0)
    {
    }

    inline bool 
    stencil::floatcmp(double_type const& x1, double_type const& x2) {
      // compare two floating point numbers
      static double_type const epsilon = 1.e-8;
      if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
        // the numbers are close enough for coordinate comparison
        return true;
      } else {
        return false;
      }
    }

    inline bool 
    stencil::floatcmp_le(double_type const& x1, double_type const& x2) {
      // compare two floating point numbers
      static double_type const epsilon = 1.e-8;

      if ( x1 < x2 ) return true;

      if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
        // the numbers are close enough for coordinate comparison
        return true;
      } else {
        return false;
      }
    }

    inline bool 
    stencil::floatcmp_ge(double_type const& x1, double_type const& x2) {
      // compare two floating point numbers
      static double_type const epsilon = 1.e-8;

      if ( x1 > x2 ) return true;

      if ( x1 + epsilon >= x2 && x1 - epsilon <= x2 ) {
        // the numbers are close enough for coordinate comparison
        return true;
      } else {
        return false;
      }
    }

    inline double_type
    stencil::interp_linear(double_type y1, double_type y2,
                           double_type x, double_type x1, double_type x2) {
      double_type xx1 = x - x1;
      double_type xx2 = x - x2;
      double_type result = xx2*y1/( (x1-x2) ) + xx1*y2/( (x2-x1) );

      return result;
    }

    // interp3d {{{
    void stencil::interp3d(double_type &x,double_type &y, double_type &z,
                           double_type minx, double_type miny,double_type minz,
                           double_type h, int nx, int ny, int nz,
                           access_memory_block<stencil_data> &val, 
                           nodedata &result, parameter const& par) {
      // set up index bounds
      int ii = (int) ( (x-minx)/h );
      int jj = (int) ( (y-miny)/h );
      int kk = (int) ( (z-minz)/h );

      const int num_eqns = HPX_SMP_AMR3D_NUM_EQUATIONS;

      bool no_interp_x = false;
      bool no_interp_y = false;
      bool no_interp_z = false;
      if ( floatcmp(h*ii + minx,x) ) {
        no_interp_x = true;
      }
      if ( floatcmp(h*jj+miny,y) ) {
        no_interp_y = true;
      }
      if ( floatcmp(h*kk+minz,z) ) {
        no_interp_z = true;
      }

      if ( no_interp_x && no_interp_y && no_interp_z ) {
        // no interp needed
        for (int ll=0;ll<num_eqns;++ll) {
          result.phi[0][ll] = val->value_[ii+nx*(jj+ny*kk)].phi[0][ll];
        }
        result.error = val->value_[ii+nx*(jj+ny*kk)].error;
        return;
      }

      // Quick sanity check to be sure we have bracketed the point we wish to interpolate
      if ( !no_interp_x ) {
        BOOST_ASSERT(floatcmp_le(h*ii+minx,x) && floatcmp_ge(h*(ii+1)+minx,x) );
      }
      if ( !no_interp_y ) {
        BOOST_ASSERT(floatcmp_le(h*jj+miny,y) && floatcmp_ge(h*(jj+1)+miny,y) );
      }
      if ( !no_interp_z ) {
        BOOST_ASSERT(floatcmp_le(h*kk+minz,z) && floatcmp_ge(h*(kk+1)+minz,z) );
      }

      double_type tmp2[2][2][HPX_SMP_AMR3D_NUM_EQUATIONS];
      double_type tmp3[2][HPX_SMP_AMR3D_NUM_EQUATIONS];
      double_type tmp2_r[2][2];
      double_type tmp3_r[2];

      // interpolate in x {{{
      if ( !no_interp_x && !no_interp_y && !no_interp_z ) {
        for (int k=kk;k<kk+2;++k) {
          for (int j=jj;j<jj+2;++j) {
            for (int ll=0;ll<num_eqns;++ll) {
              tmp2[j-jj][k-kk][ll] = interp_linear(val->value_[ii   +nx*(j+ny*k)].phi[0][ll],
                                                   val->value_[ii+1 +nx*(j+ny*k)].phi[0][ll],
                                                   x,
                                                   h*ii+minx,h*(ii+1)+minx);
            }
            tmp2_r[j-jj][k-kk] = interp_linear(val->value_[ii   +nx*(j+ny*k)].error,
                                               val->value_[ii+1 +nx*(j+ny*k)].error,
                                               x,
                                               h*ii+minx,h*(ii+1)+minx);
          }
        }
      } else if ( no_interp_x && !no_interp_y && !no_interp_z ) {
        for (int k=kk;k<kk+2;++k) {
          for (int j=jj;j<jj+2;++j) {
            for (int ll=0;ll<num_eqns;++ll) {
              tmp2[j-jj][k-kk][ll] = val->value_[ii+nx*(j+ny*k)].phi[0][ll];
            }
            tmp2_r[j-jj][k-kk] = val->value_[ii+nx*(j+ny*k)].error;
          }
        }
      } else if ( !no_interp_x && no_interp_y && !no_interp_z ) {
        for (int k=kk;k<kk+2;++k) {
          for (int ll=0;ll<num_eqns;++ll) {
            tmp2[0][k-kk][ll] = interp_linear(val->value_[ii  +nx*(jj+ny*k)].phi[0][ll],
                                              val->value_[ii+1+nx*(jj+ny*k)].phi[0][ll],
                                              x,
                                              h*ii+minx,h*(ii+1)+minx);
          }
          tmp2_r[0][k-kk] = interp_linear(val->value_[ii  +nx*(jj+ny*k)].error,
                                          val->value_[ii+1+nx*(jj+ny*k)].error,
                                          x,
                                          h*ii+minx,h*(ii+1)+minx);
        }
      } else if ( !no_interp_x && !no_interp_y && no_interp_z ) {
        for (int j=jj;j<jj+2;++j) {
          for (int ll=0;ll<num_eqns;++ll) {
            tmp2[j-jj][0][ll] = interp_linear(val->value_[ii  +nx*(j+ny*kk)].phi[0][ll],
                                              val->value_[ii+1+nx*(j+ny*kk)].phi[0][ll],
                                              x,
                                              h*ii+minx,h*(ii+1)+minx);
          }
          tmp2_r[j-jj][0] = interp_linear(val->value_[ii  +nx*(j+ny*kk)].error,
                                          val->value_[ii+1+nx*(j+ny*kk)].error,
                                          x,
                                          h*ii+minx,h*(ii+1)+minx);
        }
      } else if ( no_interp_x && no_interp_y && !no_interp_z ) {
        for (int k=kk;k<kk+2;++k) {
          for (int ll=0;ll<num_eqns;++ll) {
            tmp2[0][k-kk][ll] = val->value_[ii+nx*(jj+ny*k)].phi[0][ll];
          }
          tmp2_r[0][k-kk] = val->value_[ii+nx*(jj+ny*k)].error;
        }
      } else if ( no_interp_x && !no_interp_y && no_interp_z ) {
        for (int j=jj;j<jj+2;++j) {
          for (int ll=0;ll<num_eqns;++ll) {
            tmp2[j-jj][0][ll] = val->value_[ii+nx*(j+ny*kk)].phi[0][ll];
          }
          tmp2_r[j-jj][0] = val->value_[ii+nx*(j+ny*kk)].error;
        }
      } else if ( !no_interp_x && no_interp_y && no_interp_z ) {
        for (int ll=0;ll<num_eqns;++ll) {
          result.phi[0][ll] = interp_linear(val->value_[ii  +nx*(jj+ny*kk)].phi[0][ll],
                                            val->value_[ii+1+nx*(jj+ny*kk)].phi[0][ll],
                                            x,
                                            h*ii+minx,h*(ii+1)+minx);
        }
        result.error = interp_linear(val->value_[ii  +nx*(jj+ny*kk)].error,
                                     val->value_[ii+1+nx*(jj+ny*kk)].error,
                                     x,
                                     h*ii+minx,h*(ii+1)+minx);
        return;
      } else {
        BOOST_ASSERT(false);
      }
      // }}}

      // interpolate in y {{{
      if ( !no_interp_y && !no_interp_z ) {
        for (int k=0;k<2;++k) {
          for (int ll=0;ll<num_eqns;++ll) {
            tmp3[k][ll] = interp_linear(tmp2[0][k][ll],tmp2[1][k][ll],y,
                                         h*jj+miny,h*(jj+1)+miny);
          }
          tmp3_r[k] = interp_linear(tmp2_r[0][k],tmp2_r[1][k],y,
                                     h*jj+miny,h*(jj+1)+miny);
        }
      } else if ( no_interp_y && !no_interp_z ) {
        for (int k=0;k<2;++k) {
          for (int ll=0;ll<num_eqns;++ll) {
            tmp3[k][ll] = tmp2[0][k][ll];
          }
          tmp3_r[k] = tmp2_r[0][k];
        }
      } else if ( !no_interp_y && no_interp_z ) {
        for (int ll=0;ll<num_eqns;++ll) {
          result.phi[0][ll] = interp_linear(tmp2[0][0][ll],tmp2[1][0][ll],y,
                                                              h*jj+miny,h*(jj+1)+miny);
        }
        result.error = interp_linear(tmp2_r[0][0],tmp2_r[1][0],y,
                                     h*jj+miny,h*(jj+1)+miny);
        return;
      } else {
        BOOST_ASSERT(false);
      }
      // }}}

      // interpolate in z {{{
      if ( !no_interp_z ) {
        for (int ll=0;ll<num_eqns;++ll) {
          result.phi[0][ll] = interp_linear(tmp3[0][ll],tmp3[1][ll],
                                                              z,
                                                              h*kk+minz,h*(kk+1)+minz);
        } 
        result.error = interp_linear(tmp3_r[0],tmp3_r[1],
                                     z,
                                     h*kk+minz,h*(kk+1)+minz);
        return;
      } else {
        BOOST_ASSERT(false);
      }
      // }}}

      return;
    }
    // }}}

#if defined(SDF_FOUND)
    // write_sdf {{{
    void stencil::write_sdf(int gi,double datatime,int locality,
                            std::vector<nodedata> &value,parameter const& par ) {
      std::vector<double> x,y,z,phi,d1phi,d2phi,d3phi,d4phi;   

      int nx = par->gr_nx[gi];
      int ny = par->gr_ny[gi];
      int nz = par->gr_nz[gi];
      for (int i=0;i<nx;i++) {
        x.push_back(par->gr_minx[gi] + par->gr_h[gi]*i);
      }
      for (int i=0;i<ny;i++) {
        x.push_back(par->gr_miny[gi] + par->gr_h[gi]*i);
      }
      for (int i=0;i<nz;i++) {
        x.push_back(par->gr_minz[gi] + par->gr_h[gi]*i);
      }

      for (int k=0;k<nz;k++) {
      for (int j=0;j<ny;j++) {
      for (int i=0;i<nx;i++) {
        phi.push_back(value[i+nx*(j+ny*k)].phi[0][0]);
        d1phi.push_back(value[i+nx*(j+ny*k)].phi[0][1]);
        d2phi.push_back(value[i+nx*(j+ny*k)].phi[0][2]);
        d3phi.push_back(value[i+nx*(j+ny*k)].phi[0][3]);
        d4phi.push_back(value[i+nx*(j+ny*k)].phi[0][4]);
      } } }
      int shape[3];
      char cnames[80] = { "x|y|z" };
      char phi_name[80];
      sprintf(phi_name,"%dphi_%5.3f",locality,datatime);
    //  char phi1_name[80];
    //  sprintf(phi1_name,"%dd1phi",locality);
    //  char phi2_name[80];
    //  sprintf(phi2_name,"%dd2phi",locality);
    //  char phi3_name[80];
    //  sprintf(phi3_name,"%dd3phi",locality);
    //  char phi4_name[80];
    //  sprintf(phi4_name,"%dd4phi",locality);
      shape[0] = nx;
      shape[1] = ny;
      shape[2] = nz;
      gft_out_full(phi_name,datatime,shape,cnames,3,&*x.begin(),&*phi.begin());
   //   gft_out_full(phi1_name,datatime,shape,cnames,3,&*x.begin(),&*d1phi.begin());
   //   gft_out_full(phi2_name,datatime,shape,cnames,3,&*x.begin(),&*d2phi.begin());
   //   gft_out_full(phi3_name,datatime,shape,cnames,3,&*x.begin(),&*d3phi.begin());
   //   gft_out_full(phi4_name,datatime,shape,cnames,3,&*x.begin(),&*d4phi.begin());
    }
    // }}}
#endif

    ///////////////////////////////////////////////////////////////////////////
    // Implement actual functionality of this stencil
    // Compute the result value for the current time step
    int stencil::eval(naming::id_type const& result, 
        std::vector<naming::id_type> const& gids, std::size_t row, std::size_t column,
        double cycle_time, parameter const& par)
    {
        // make sure all the gids are looking valid
        if (result == naming::invalid_id)
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "stencil::eval", "result gid is invalid");
            return -1;
        }


        // this should occur only after result has been delivered already
        BOOST_FOREACH(naming::id_type gid, gids)
        {
            if (gid == naming::invalid_id)
                return -1;
        }

        // get all input and result memory_block_data instances
        std::vector<access_memory_block<stencil_data> > val;
        access_memory_block<stencil_data> resultval = 
            get_memory_block_async(val, gids, result);

        // lock all user defined data elements, will be unlocked at function exit
        scoped_values_lock<lcos::mutex> l(resultval, val); 

        // Check if this is a prolongation/restriction step
        if ( (row+5)%3 == 0 || ( par->allowedl == 0 && row == 0 ) ) {
          // This is a prolongation/restriction step
          resultval.get() = val[0].get();
          BOOST_ASSERT(val.size() == 1);
        } else {
          //resultval->level_ = val[0]->level_;
          // Find the gi we are updating  
          std::size_t update = 365;
          for (std::size_t i=0;i<val.size();i++) {
             if ( val[i]->index_ == column ) {
               // found it
               update = i;
               break;
             }
          }
          BOOST_ASSERT(update != 365);

          resultval->max_index_ = val[update]->max_index_;
          resultval->index_ = val[update]->index_;
          resultval->level_ = val[update]->level_;
          resultval->value_.resize(val[update]->value_.size());

          int gi = par->item2gi[column];
          int nx0 = par->gr_nx[gi];
          int ny0 = par->gr_ny[gi];
          int nz0 = par->gr_nz[gi];
          double h = par->gr_h[gi];
          double minx0 = par->gr_minx[gi];
          double miny0 = par->gr_miny[gi];
          double minz0 = par->gr_minz[gi];
          double t = val[0]->timestep_*par->h*par->lambda + cycle_time;

#if 0      
          // local array ( to help with book-keeping )
          int lnx = nx0+6;
          int lny = ny0+6;
          int lnz = nz0+6;
          std::vector<int> src(lnx*lny*lnz,-1);
          std::vector<int> vsrc(lnx*lny*lnz,-1);

          std::size_t oii = 0;
          int ogi = par->item2gi[ val[oii]->index_ ];
          bool proceed;
          for (int k=0;k<lnz;k++) {
            double z = minz0 + (k-3)*h;
          for (int j=0;j<lny;j++) {
            double y = miny0 + (j-3)*h;
          for (int i=0;i<lnx;i++) {
            double x = minx0 + (i-3)*h;
            if ( i >= 3 && i < nx0 + 3 &&
                 j >= 3 && j < ny0 + 3 &&
                 k >= 3 && k < nz0 + 3 ) {
              //src[i + lnx*(j+lny*k)] = val[update]->value_[i-3 + nx0*(j-3 + ny0*(k-3))];
              src[i + lnx*(j+lny*k)] = i-3 + nx0*(j-3 + ny0*(k-3));
                // TEST
                if ( src[i + lnx*(j+lny*k)] < 0 ) {
                  std::cout << " src PROBLEM A: " << src[i + lnx*(j+lny*k)] << " vsrc " << oii << std::endl;
                  BOOST_ASSERT(false);
                }
                // END TEST
              vsrc[i + lnx*(j+lny*k)] = update;
            } else {
              // this data point comes from the communicated neighbors
              // but you have to figure out which

              proceed = false;
              if ( x <= par->gr_maxx[ogi] && x >= par->gr_minx[ogi] &&
                   y <= par->gr_maxy[ogi] && y >= par->gr_miny[ogi] &&
                   z <= par->gr_maxz[ogi] && z >= par->gr_minz[ogi]
                 ) {
                proceed = true;
              } else {
                for (std::size_t ii=0;ii<val.size();ii++) {
                  if ( ii != update ) {
                    int gi2 = par->item2gi[ val[ii]->index_ ];
                    // check if x,y,z is in this mesh
                    if ( x <= par->gr_maxx[gi2] && x >= par->gr_minx[gi2] &&
                         y <= par->gr_maxy[gi2] && y >= par->gr_miny[gi2] &&
                         z <= par->gr_maxz[gi2] && z >= par->gr_minz[gi2] ) {
                      // found
                      ogi = gi2;
                      oii = ii;
                      proceed = true;
                      break;
                    }
                  }
                }
              }
              if ( proceed ) {
                // find the index
                double llh = par->gr_h[ogi];
                int llnx = par->gr_nx[ogi];
                int llny = par->gr_ny[ogi];
                int ii = (int) ( (x - par->gr_minx[ogi])/llh ); 
                int jj = (int) ( (y - par->gr_miny[ogi])/llh ); 
                int kk = (int) ( (z - par->gr_minz[ogi])/llh ); 
               // src[i + lnx*(j+lny*k)] = val[ oii ]->value_[ii + llnx*(jj + llny*kk)];
                src[i + lnx*(j+lny*k)] = ii + llnx*(jj + llny*kk);
                BOOST_ASSERT(src[i + lnx*(j+lny*k)] >= 0 );
                vsrc[i + lnx*(j+lny*k)] = oii;
              }
            }
          } } }

          // call rk update
          rkupdate(val,resultval.get_ptr(),src,vsrc,par->lambda*h,h,t,
                   nx0,ny0,nz0,minx0,miny0,minz0,*par.p); 
#endif

          resultval->timestep_ = val[0]->timestep_ + 1.0/pow(2.0,int(val[0]->level_));
//#if 0
          // TEST
          t = resultval->timestep_*par->h*par->lambda + cycle_time;
          for (int k=0;k<nz0;k++) {
            double z = minz0 + k*h;
          for (int j=0;j<ny0;j++) {
            double y = miny0 + j*h;
          for (int i=0;i<nx0;i++) {
            double x = minx0 + i*h;

            // Provide initial error
            double d = 11.0;
            double A = -(x-0.5*d*cos(t))*(x-0.5*d*cos(t)) - (y+0.5*d*sin(t))*(y+0.5*d*sin(t)) - z*z;        
            double B = -(x+0.5*d*cos(t))*(x+0.5*d*cos(t)) - (y-0.5*d*sin(t))*(y-0.5*d*sin(t)) - z*z;        
            double Phi = exp(A) + exp(B);
            resultval->value_[i+nx0*(j+ny0*k)].error = Phi;
            resultval->value_[i+nx0*(j+ny0*k)].phi[0][0] = Phi;
          } } }
//#endif
#if defined(RNPL_FOUND)
          // Output
          if (par->loglevel > 1 && fmod(resultval->timestep_, par->output) < 1.e-6) {
            std::vector<double> x,y,z,phi,d1phi,d2phi,d3phi,d4phi;   
            applier::applier& appl = applier::get_applier();
            naming::id_type this_prefix = appl.get_runtime_support_gid();
            int locality = get_prefix_from_id( this_prefix );
            double datatime = resultval->timestep_*par->h*par->lambda + cycle_time;
            write_sdf(gi,datatime,locality,resultval->value_,par);
          }
#endif
        }

        //std::cout << " TEST row " << row << " column " << column << " timestep " << resultval->timestep_ << std::endl;
        if ( val[0]->timestep_ >= par->refine_every-2 ) {
          return 0;
        }
        return 1;
    }

    hpx::actions::manage_object_action<stencil_data> const manage_stencil_data =
        hpx::actions::manage_object_action<stencil_data>();

}}}

HPX_REGISTER_MANAGE_OBJECT_ACTION(
    hpx::actions::manage_object_action<hpx::components::amr::stencil_data>,
    marduk_manage_object_action_stencil_data)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    ///////////////////////////////////////////////////////////////////////////
    naming::id_type stencil::alloc_data(int item, int maxitems, int row,
                           std::vector<naming::id_type> const& interp_src_data,
                           double time,
                           parameter const& par)
    {
        naming::id_type here = applier::get_applier().get_runtime_support_gid();
        naming::id_type result = components::stubs::memory_block::create(
            here, sizeof(stencil_data), manage_stencil_data);

        if (-1 != item) {
            // provide initial data for the given data value 
            access_memory_block<stencil_data> val(
                components::stubs::memory_block::checkout(result));

            if ( time < 1.e-8 ) {
              // call provided (external) function
              generate_initial_data(val.get_ptr(), item, maxitems, row, *par.p);
            } else {
              // data is generated from interpolation using interp_src_data
              // find the bounding box for this item
              double minx = par->gr_minx[par->item2gi[item]];
              double miny = par->gr_miny[par->item2gi[item]];
              double minz = par->gr_minz[par->item2gi[item]];
              double maxx = par->gr_maxx[par->item2gi[item]];
              double maxy = par->gr_maxy[par->item2gi[item]];
              double maxz = par->gr_maxz[par->item2gi[item]];
              double h = par->gr_h[par->item2gi[item]];
              int nx0 = par->gr_nx[par->item2gi[item]];
              int ny0 = par->gr_ny[par->item2gi[item]];
              int nz0 = par->gr_nz[par->item2gi[item]];

              bool complete = false;
              std::vector<int> vindex(nx0*ny0*nz0,-1);

              val->max_index_ = maxitems;
              val->index_ = item;
              val->timestep_ = 0;
              val->value_.resize(nx0*ny0*nz0);
              for (std::size_t step=0;step<par->prev_gi.size();step++) {
                // see if the new gi is the same as the old
                int gi = par->prev_gi[step];
                if ( gi != -1 ) {
                  if ( floatcmp(minx,par->gr_minx[gi]) && 
                       floatcmp(miny,par->gr_miny[gi]) && 
                       floatcmp(minz,par->gr_minz[gi]) && 
                       floatcmp(maxx,par->gr_maxx[gi]) && 
                       floatcmp(maxy,par->gr_maxy[gi]) && 
                       floatcmp(maxz,par->gr_maxz[gi]) && 
                       floatcmp(h,par->gr_h[gi]) 
                     ) 
                  {
                    // This has the same data -- copy it over
                    access_memory_block<stencil_data> prev_val(
                      components::stubs::memory_block::checkout(interp_src_data[par->prev_gi2item[gi]]));
                    val.get() = prev_val.get();

                    val->max_index_ = maxitems;
                    val->index_ = item;
                    val->timestep_ = 0;
                    complete = true;
                    break;
                  } else if (
                    intersection(minx,maxx,
                                 miny,maxy,
                                 minz,maxz,
                                 par->gr_minx[gi],par->gr_maxx[gi],
                                 par->gr_miny[gi],par->gr_maxy[gi],
                                 par->gr_minz[gi],par->gr_maxz[gi])
                    && floatcmp(h,par->gr_h[gi]) 
                            ) 
                  {
                    access_memory_block<stencil_data> prev_val(
                      components::stubs::memory_block::checkout(interp_src_data[par->prev_gi2item[gi]]));
                    // find the intersection index
                    double x1 = (std::max)(minx,par->gr_minx[gi]); 
                    double x2 = (std::min)(maxx,par->gr_maxx[gi]); 
                    double y1 = (std::max)(miny,par->gr_miny[gi]); 
                    double y2 = (std::min)(maxy,par->gr_maxy[gi]); 
                    double z1 = (std::max)(minz,par->gr_minz[gi]); 
                    double z2 = (std::min)(maxz,par->gr_maxz[gi]);

                    int isize = (int) ( (x2-x1)/h );
                    int jsize = (int) ( (y2-y1)/h );
                    int ksize = (int) ( (z2-z1)/h );

                    int lnx = par->gr_nx[gi]; 
                    int lny = par->gr_ny[gi]; 
//                    int lnz = par->gr_nz[gi]; 

                    int istart_dst = (int) ( (x1 - minx)/h );
                    int jstart_dst = (int) ( (y1 - miny)/h );
                    int kstart_dst = (int) ( (z1 - minz)/h );

                    int istart_src = (int) ( (x1 - par->gr_minx[gi])/h );
                    int jstart_src = (int) ( (y1 - par->gr_miny[gi])/h );
                    int kstart_src = (int) ( (z1 - par->gr_minz[gi])/h );

                    val->level_ = prev_val->level_;
                    for (int kk=0;kk<=ksize;kk++) {
                    for (int jj=0;jj<=jsize;jj++) {
                    for (int ii=0;ii<=isize;ii++) {
                      int i = ii + istart_dst;
                      int j = jj + jstart_dst;
                      int k = kk + kstart_dst;

                      int si = ii + istart_src;
                      int sj = jj + jstart_src;
                      int sk = kk + kstart_src;
                      BOOST_ASSERT(i+nx0*(j+ny0*k) < int(val->value_.size()));
                      BOOST_ASSERT(si+lnx*(sj+lny*sk) < int(prev_val->value_.size()));
                      val->value_[i+nx0*(j+ny0*k)] = prev_val->value_[si+lnx*(sj+lny*sk)];

                      // record that the value at this index doesn't need interpolation
                      vindex[i+nx0*(j+ny0*k)] = 1;
                    } } }

                    // record what indices in the dst were filled in
                    complete = false;
                  }
                }
              }

              if ( complete == false ) {
                // some interpolation from a lower resolution mesh is required
                // fill in the blanks

                // initialize ogi to possibly save a for loop
                int ogi = par->prev_gi[0];
                BOOST_ASSERT(ogi != -1);
                bool proceed = false;;
                for (int kk=0;kk<nz0;kk++) {
                  double z = minz + kk*h;
                for (int jj=0;jj<ny0;jj++) {
                  double y = miny + jj*h;
                for (int ii=0;ii<nx0;ii++) {
                  double x = minx + ii*h;

                  // is this a point that needs interpolation?
                  if ( vindex[ii+nx0*(jj+ny0*kk)] == -1 ) {
                    // interp needed -- 
                    proceed = false;
                    // try to avoid the inner loop
                    if (floatcmp(h,par->refine_factor*par->gr_h[ogi]) &&
                         x >= par->gr_minx[ogi] &&
                         x <= par->gr_maxx[ogi] &&
                         y >= par->gr_miny[ogi] &&
                         y <= par->gr_maxy[ogi] &&
                         z >= par->gr_minz[ogi] &&
                         z <= par->gr_maxz[ogi] )
                    {
                      proceed = true;
                    } else {
                      for (std::size_t cycle=1;cycle <= val->level_;cycle++) {
                        for (std::size_t step=0;step<par->prev_gi.size();step++) {
                          int gi = par->prev_gi[step];
                          if ( gi != -1 ) {
                            if (floatcmp(pow((double)par->refine_factor,(int)cycle)*h,par->gr_h[gi]) &&
                               x >= par->gr_minx[gi] &&
                               x <= par->gr_maxx[gi] &&
                               y >= par->gr_miny[gi] &&
                               y <= par->gr_maxy[gi] &&
                               z >= par->gr_minz[gi] &&
                               z <= par->gr_maxz[gi] ) 
                            {
                              ogi = gi;
                              proceed = true;
                              break;
                            }
                          }
                        }
                        if (proceed) break;
                      }
                    }

                    if ( proceed ) {
                      access_memory_block<stencil_data> prev_val(
                                 components::stubs::memory_block::checkout(
                                          interp_src_data[par->prev_gi2item[ogi]]));
                      // interpolate
                      interp3d(x,y,z,
                               par->gr_minx[ogi],par->gr_miny[ogi],par->gr_minz[ogi],
                               par->gr_h[ogi],par->gr_nx[ogi],par->gr_ny[ogi],par->gr_nz[ogi],
                               prev_val,val->value_[ii+nx0*(jj+ny0*kk)],par);
                    } else {
                      std::cout << " interp anchor not found! PROBLEM " << std::endl;
                      std::cout << " Looking for point x " << x << " y " << y << " z " << z << std::endl;
                      std::cout << " Here's what we have to choose from " << std::endl;
                      for (std::size_t step=0;step<par->prev_gi.size();step++) {
                        int gi = par->prev_gi[step];
                        if ( gi != -1 ) { 
                          std::cout << " bbox: " << par->gr_minx[gi] << " " << par->gr_maxx[gi] << std::endl;
                          std::cout << "       " << par->gr_miny[gi] << " " << par->gr_maxy[gi] << std::endl;
                          std::cout << "       " << par->gr_minz[gi] << " " << par->gr_maxz[gi] << std::endl;
                          std::cout << "  h    " << par->gr_h[gi] << " our level h " << h << " *2 " << par->refine_factor*h << std::endl;
                        }
                      }
                      BOOST_ASSERT(false);
                    }

                  } 
                } } }

              }

              //std::cout << " TEST bbox in gen init data " << xmin << " " << xmax << std::endl;
              //std::cout << "                            " << ymin << " " << ymax << std::endl;
              //std::cout << "                            " << zmin << " " << zmax << std::endl;

              // TEST
              //generate_initial_data(val.get_ptr(), item, maxitems, row, *par.p);
               
            }

            if (log_ && par->loglevel > 1)         // send initial value to logging instance
                stubs::logging::logentry(log_, val.get(), row,0, par);
        }
        return result;
    }

    void stencil::init(std::size_t numsteps, naming::id_type const& logging)
    {
        numsteps_ = numsteps;
        log_ = logging;
    }

}}}

