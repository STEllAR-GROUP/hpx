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
    ///////////////////////////////////////////////////////////////////////////
    // Implement actual functionality of this stencil
    // Compute the result value for the current time step
    void logging::logentry(stencil_data const& val, int row, int logcode, Parameter const& par)
    {
        mutex_type::scoped_lock l(mtx_);

        if ( par.output_stdout == 1 && ( val.iter_ == 0 || par.integrator == 0 ) ) {
          std::cout << " AMR Level: " << val.level_ << "   Timestep: " <<  val.timestep_ << "   refine?: " << val.refine_ << "   row: " << row << "   index: " << val.index_ << "    Value: " << val.value_.phi0 << "  x-coordinate : " << val.x_ << std::endl;
         // if (val.right_alloc_ == 1) {
         //   std::cout << " right value : " << val.right_value_ << " right level : " << val.right_level_ << " right alloc : " << val.right_alloc_ << std::endl;   
         // } 
        }

        // output to file "output.dat"
        FILE *fdata;
        if ( logcode == 0 && ( val.iter_ == 0 || par.integrator == 0 ) ) {
          fdata = fopen("output.dat","a");
          fprintf(fdata,"%d %g %g %g\n",val.level_,val.timestep_,val.x_,val.value_.phi0);
          fclose(fdata);
        }

        // Debugging measures
        // output file to "logcode1.dat"
        if ( logcode == 1 ) {
          fdata = fopen("logcode1.dat","a");
          fprintf(fdata,"%d %g %g %g\n",val.level_,val.timestep_,val.x_,val.value_.phi0);
          fclose(fdata);
        }
        //
        // output file to "logcode2.dat"
        if ( logcode == 2 ) {
          fdata = fopen("logcode2.dat","a");
          fprintf(fdata,"%d %g %g %g\n",val.level_,val.timestep_,val.x_,val.value_.phi0);
          fclose(fdata);
        }
    }

}}}}

