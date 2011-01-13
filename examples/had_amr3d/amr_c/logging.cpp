//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <hpx/hpx.hpp>

#if defined(SDF_FOUND)
#include <sdf.h>
#endif

#include "logging.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server
{
    logging::mutex_type logging::mtx_("logging");

    inline std::string convert(double d)
    {
      return boost::lexical_cast<std::string>(d);
    }

#if MPFR_FOUND != 0
    inline std::string convert(mpfr::mpreal const & d)
    {
      return d.to_string();
    }
#endif


    ///////////////////////////////////////////////////////////////////////////
    // Implement actual functionality of this stencil
    // Compute the result value for the current time step
    void logging::logentry(stencil_data const& val, int row, int logcode, Parameter const& par)
    {
        mutex_type::scoped_lock l(mtx_);
        int i;

        if ( par->output_stdout == 1 ) {
          if (fmod(val.timestep_,par->output) < 1.e-6) {
           // for (i=0;i<val.granularity*val.granularity*val.granularity;i++) {
              i = 0;
              std::cout << " AMR Level: " << val.level_ 
                        << " Timestep: " <<  val.timestep_ 
                        << " Time: " << val.timestep_*par->dx0*par->lambda  
                        << " row: " << row 
                        << " index: " << val.index_ 
                        << " Value: " << val.value_[i].phi[0][0] 
                        << " x: " << val.x_[i] << " y: " << val.y_[i] << " z: " << val.z_[i] 
                        << std::endl << std::flush ;
          //  }
          }
        }

        // output to file "output.dat"
        FILE *fdata;
        std::vector<double> x,y,z,Phi,chi,Pi;
        double datatime;
        if ( logcode == 0 ) {
          if (fmod(val.timestep_,par->output) < 1.e-6 && val.level_ >= par->output_level) {
            for (i=0;i<val.granularity;i++) {
              x.push_back(val.x_[0]+i*par->dx0);
            }
            for (i=0;i<val.granularity;i++) {
              x.push_back(val.y_[0]+i*par->dx0);
            }
            for (i=0;i<val.granularity;i++) {
              x.push_back(val.z_[0]+i*par->dx0);
            }
            for (i=0;i<val.granularity*val.granularity*val.granularity;i++) {
              chi.push_back(val.value_[i].phi[0][0]);
              Phi.push_back(val.value_[i].phi[0][4]);
              datatime = val.timestep_*par->dx0*par->lambda;
#if 0
              std::string x_str = convert(val.x_[i]);
              std::string chi_str = convert(val.value_[i].phi[0][0]);
              std::string Phi_str = convert(val.value_[i].phi[0][1]);
              std::string Pi_str = convert(val.value_[i].phi[0][2]);
              std::string time_str = convert(val.timestep_*par->dx0*par->lambda);

              fdata = fopen("chi.dat","a");
              fprintf(fdata,"%d %s %s %s\n",val.level_,time_str.c_str(),x_str.c_str(),chi_str.c_str());
              fclose(fdata);

              fdata = fopen("Phi.dat","a");
              fprintf(fdata,"%d %s %s %s\n",val.level_,time_str.c_str(),x_str.c_str(),Phi_str.c_str());
              fclose(fdata);

              fdata = fopen("Pi.dat","a");
              fprintf(fdata,"%d %s %s %s\n",val.level_,time_str.c_str(),x_str.c_str(),Pi_str.c_str());
              fclose(fdata);

#endif
            }
#if defined(SDF_FOUND)
            int shape[3];
            char cnames[80] = { "x|y|z" };
            shape[0] = par->granularity;
            shape[1] = par->granularity;
            shape[2] = par->granularity;
            gft_out_full("phi",datatime,shape,cnames,3,&*x.begin(),&*chi.begin());
            gft_out_full("d4phi",datatime,shape,cnames,3,&*x.begin(),&*Phi.begin());
#endif
          }
        }
    }

}}}}

