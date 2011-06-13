//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Matthew Anderson
//  Copyright (c)      2011 Bryce Lelbach
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/lcos/future_wait.hpp>

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

bool intersection(double xmin,double xmax,
                  double ymin,double ymax,
                  double zmin,double zmax,
                  double xmin2,double xmax2,
                  double ymin2,double ymax2,
                  double zmin2,double zmax2);
double max(double,double);
double min(double,double);

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

      int num_eqns = HPX_SMP_AMR3D_NUM_EQUATIONS;

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

      double_type tmp2[2][2][num_eqns];
      double_type tmp3[2][num_eqns];
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

        resultval.get() = val[0].get();
        resultval->level_ = val[0]->level_;

        // Check if this is a prolongation/restriction step
        if ( (row+5)%3 == 0 || ( par->allowedl == 0 && row == 0 ) ) {
          // This is a prolongation/restriction step
          resultval->timestep_ = val[0]->timestep_;
        } else {
          resultval->timestep_ = val[0]->timestep_ + 1.0/pow(2.0,int(val[0]->level_));

#if defined(RNPL_FOUND)
          // Output
          if (par->loglevel > 1 && fmod(resultval->timestep_, par->output) < 1.e-6) {
            std::vector<double> x,y,z,phi,d1phi,d2phi,d3phi,d4phi;   
            applier::applier& appl = applier::get_applier();
            naming::id_type this_prefix = appl.get_runtime_support_gid();
            int locality = get_prefix_from_id( this_prefix );
            double datatime;

            int gi = par->item2gi[column];
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

            datatime = resultval->timestep_*par->h*par->lambda + cycle_time;
            //datatime = resultval->timestep_;

            //std::cout << " TEST datatime " << datatime << " nx " << nx << " ny " << ny << " nz " << nz << " minx " << par->gr_minx[gi] << " miny " << par->gr_miny[gi] << " minz " << par->gr_minz[gi] << " maxx " << par->gr_maxx[gi] << " maxy " << par->gr_maxy[gi] << " maxz " << par->gr_maxz[gi] << std::endl;

            for (int k=0;k<nz;k++) {
            for (int j=0;j<ny;j++) {
            for (int i=0;i<nx;i++) {
              phi.push_back(resultval->value_[i+nx*(j+ny*k)].phi[0][0]);
              d1phi.push_back(resultval->value_[i+nx*(j+ny*k)].phi[0][1]);
              d2phi.push_back(resultval->value_[i+nx*(j+ny*k)].phi[0][2]);
              d3phi.push_back(resultval->value_[i+nx*(j+ny*k)].phi[0][3]);
              d4phi.push_back(resultval->value_[i+nx*(j+ny*k)].phi[0][4]);
            } } }
            int shape[3];
            char cnames[80] = { "x|y|z" };
            char phi_name[80];
            sprintf(phi_name,"%dphi",locality);
            char phi1_name[80];
            sprintf(phi1_name,"%dd1phi",locality);
            char phi2_name[80];
            sprintf(phi2_name,"%dd2phi",locality);
            char phi3_name[80];
            sprintf(phi3_name,"%dd3phi",locality);
            char phi4_name[80];
            sprintf(phi4_name,"%dd4phi",locality);
            shape[0] = nx;
            shape[1] = ny;
            shape[2] = nz;
            gft_out_full(phi_name,datatime,shape,cnames,3,&*x.begin(),&*phi.begin());
            gft_out_full(phi1_name,datatime,shape,cnames,3,&*x.begin(),&*d1phi.begin());
            gft_out_full(phi2_name,datatime,shape,cnames,3,&*x.begin(),&*d2phi.begin());
            gft_out_full(phi3_name,datatime,shape,cnames,3,&*x.begin(),&*d3phi.begin());
            gft_out_full(phi4_name,datatime,shape,cnames,3,&*x.begin(),&*d4phi.begin());
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
                  double x1 = max(minx,par->gr_minx[gi]); 
                  double x2 = min(maxx,par->gr_maxx[gi]); 
                  double y1 = max(miny,par->gr_miny[gi]); 
                  double y2 = min(maxy,par->gr_maxy[gi]); 
                  double z1 = max(minz,par->gr_minz[gi]); 
                  double z2 = min(maxz,par->gr_maxz[gi]);

                  int isize = (int) ( (x2-x1)/h );
                  int jsize = (int) ( (y2-y1)/h );
                  int ksize = (int) ( (z2-z1)/h );

                  int lnx = par->gr_nx[gi]; 
                  int lny = par->gr_ny[gi]; 
                  int lnz = par->gr_nz[gi]; 

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
                    BOOST_ASSERT(i+nx0*(j+ny0*k) < val->value_.size());
                    BOOST_ASSERT(si+lnx*(sj+lny*sk) < prev_val->value_.size());
                    val->value_[i+nx0*(j+ny0*k)] = prev_val->value_[si+lnx*(sj+lny*sk)];

                    // record that the value at this index doesn't need interpolation
                    vindex[i+nx0*(j+ny0*k)] = 1;
                  } } }

                  // record what indices in the dst were filled in
                  complete = false;
                }
              }

              if ( complete == false ) {
                // some interpolation from a lower resolution mesh is required
                // fill in the blanks

                // initialize ogi to possibly save a for loop
                int ogi = par->prev_gi[0];
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
                    // try to aviod the inner loop
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
                          if (floatcmp(pow(par->refine_factor,cycle)*h,par->gr_h[gi]) &&
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
                        std::cout << " bbox: " << par->gr_minx[gi] << " " << par->gr_maxx[gi] << std::endl;
                        std::cout << "       " << par->gr_miny[gi] << " " << par->gr_maxy[gi] << std::endl;
                        std::cout << "       " << par->gr_minz[gi] << " " << par->gr_maxz[gi] << std::endl;
                        std::cout << "  h    " << par->gr_h[gi] << " our level h " << h << " *2 " << par->refine_factor*h << std::endl;
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

