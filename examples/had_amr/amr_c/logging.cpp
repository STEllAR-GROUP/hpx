//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <hpx/hpx.hpp>

#include "logging.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server
{
    logging::mutex_type logging::mtx_("logging");

    ///////////////////////////////////////////////////////////////////////////
    // Implement actual functionality of this stencil
    // Compute the result value for the current time step
    void logging::logentry(stencil_data const& val, int row, int logcode, Parameter const& par)
    {
        mutex_type::scoped_lock l(mtx_);

        if ( par.output_stdout == 1 && val.iter_ == 0 ) {
          if (fmod(val.timestep_,par.output) < 1.e-6) {
            std::cout << " AMR Level: " << val.level_ 
                      << " Timestep: " <<  val.timestep_ 
                      << " Time: " << val.timestep_*par.dx0*par.lambda  
                      << " refine?: " << val.refine_ 
                      << " row: " << row 
                      << " index: " << val.index_ 
                      << " Value: " << val.value_.phi[0][0] 
                      << " x-coordinate: " << val.x_ 
                      << std::endl << std::flush ;
          }
        }

        // output to file "output.dat"
        FILE *fdata;
        if ( logcode == 0 && val.iter_ == 0 ) {
          if (fmod(val.timestep_,par.output) < 1.e-6 && val.x_ >= 0.0 && val.level_ >= par.output_level) {
            std::string x_str = boost::lexical_cast<std::string>(val.x_);
            std::string chi_str = boost::lexical_cast<std::string>(val.value_.phi[0][0]);
            std::string Phi_str = boost::lexical_cast<std::string>(val.value_.phi[0][1]);
            std::string Pi_str = boost::lexical_cast<std::string>(val.value_.phi[0][2]);
            std::string energy_str = boost::lexical_cast<std::string>(val.value_.phi[0][3]);
            std::string time_str = boost::lexical_cast<std::string>(val.timestep_*par.dx0*par.lambda);

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
        }

        // Debugging measures
        // output file to "logcode1.dat"
        if ( logcode == 1 ) {
          std::string x_str = boost::lexical_cast<std::string>(val.x_);
          std::string chi_str = boost::lexical_cast<std::string>(val.value_.phi[0][0]);
          std::string time_str = boost::lexical_cast<std::string>(val.timestep_*par.dx0*par.lambda);

          fdata = fopen("logcode1.dat","a");
          fprintf(fdata,"%d %s %s %s\n",val.level_,time_str.c_str(),x_str.c_str(),chi_str.c_str());
          fclose(fdata);
        }
        //
        // output file to "logcode2.dat"
        if ( logcode == 2 ) {
          std::string x_str = boost::lexical_cast<std::string>(val.x_);
          std::string chi_str = boost::lexical_cast<std::string>(val.value_.phi[0][0]);
          std::string time_str = boost::lexical_cast<std::string>(val.timestep_*par.dx0*par.lambda);

          fdata = fopen("logcode2.dat","a");
          fprintf(fdata,"%d %s %s %s\n",val.level_,time_str.c_str(),x_str.c_str(),chi_str.c_str());
          fclose(fdata);
        }
    }

}}}}

