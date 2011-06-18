//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <hpx/hpx.hpp>

#if defined(RNPL_FOUND)
#include <sdf.h>
#endif

#include "logging.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace nbody { namespace server
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
        //int i;

        if ( par->output_stdout == 1 ) {
//              std::cout << " X: " <<  val.x 
//                        << " Y: " << val.y  
//                        << " Z: " << val.z  
//                        << " row: " << val.row  
//                        << " column: " << val.column  << std::endl;
        }
    }

}}}}

