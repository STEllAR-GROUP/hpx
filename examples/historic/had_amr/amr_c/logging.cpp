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
            for (i=0;i<val.granularity;i++) {
              std::cout << " AMR Level: " << val.level_ 
                        << " Timestep: " <<  val.timestep_ 
                        << " Time: " << val.timestep_*par->dx0*par->lambda  
                        << " row: " << row 
                        << " index: " << val.index_ 
                        << " Value: " << val.value_[i].phi[0][0] 
                        << " x-coordinate: " << val.x_[i] 
                        << std::endl << std::flush ;
            }
          }
        }

        // output to file "output.dat"
        FILE *fdata;
        std::vector<double> x,Phi,chi,Pi,energy;
        double datatime;
        if ( logcode == 0 ) {
          if (fmod(val.timestep_,par->output) < 1.e-6 && val.level_ >= par->output_level) {
            for (i=0;i<val.granularity;i++) {
              x.push_back(val.x_[i]);
              chi.push_back(val.value_[i].phi[0][0]);
              Phi.push_back(val.value_[i].phi[0][1]);
              Pi.push_back(val.value_[i].phi[0][2]);
              energy.push_back(val.value_[i].energy);
              datatime = val.timestep_*par->dx0*par->lambda;

              std::string x_str = convert(val.x_[i]);
              std::string chi_str = convert(val.value_[i].phi[0][0]);
              std::string Phi_str = convert(val.value_[i].phi[0][1]);
              std::string Pi_str = convert(val.value_[i].phi[0][2]);
              std::string energy_str = convert(val.value_[i].energy);
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

              fdata = fopen("energy.dat","a");
              fprintf(fdata,"%d %s %s %s\n",val.level_,time_str.c_str(),x_str.c_str(),energy_str.c_str());
              fclose(fdata);
            }
#if defined(SDF_FOUND)
            int shape[3];
            char cnames[80] = { "r" };
            shape[0] = x.size(); 
            gft_out_full("chi",datatime,shape,cnames,1,&*x.begin(),&*chi.begin());
            gft_out_full("Phi",datatime,shape,cnames,1,&*x.begin(),&*Phi.begin());
            gft_out_full("Pi",datatime,shape,cnames,1,&*x.begin(),&*Pi.begin());
            gft_out_full("energy",datatime,shape,cnames,1,&*x.begin(),&*energy.begin());
#endif
          }
        }

        // Debugging measures
        // output file to "logcode1.dat"
        if ( logcode == 1 ) {
          for (i=0;i<val.granularity;i++) {
            x.push_back(val.x_[i]);
            chi.push_back(val.value_[i].phi[0][0]);
            datatime = val.timestep_*par->dx0*par->lambda;

            std::string x_str = convert(val.x_[i]);
            std::string chi_str = convert(val.value_[i].phi[0][0]);
            std::string time_str = convert(val.timestep_*par->dx0*par->lambda);

            fdata = fopen("logcode1.dat","a");
            fprintf(fdata,"%d %s %s %s\n",val.level_,time_str.c_str(),x_str.c_str(),chi_str.c_str());
            fclose(fdata);
          }
        }
        //
        // output file to "logcode2.dat"
        if ( logcode == 2 ) {
          for (i=0;i<val.granularity;i++) {
            x.push_back(val.x_[i]);
            chi.push_back(val.value_[i].phi[0][0]);
            datatime = val.timestep_*par->dx0*par->lambda;

            std::string x_str = convert(val.x_[i]);
            std::string chi_str = convert(val.value_[i].phi[0][0]);
            std::string time_str = convert(val.timestep_*par->dx0*par->lambda);

            fdata = fopen("logcode2.dat","a");
            fprintf(fdata,"%d %s %s %s\n",val.level_,time_str.c_str(),x_str.c_str(),chi_str.c_str());
            fclose(fdata);
          }
        }
    }

}}}}

