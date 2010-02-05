//  Copyright (c) 2009 Matt Anderson
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_PARAMETER_OCT_19_2009_0834AM)
#define HPX_COMPONENTS_PARAMETER_OCT_19_2009_0834AM

#include <boost/cstdint.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/serialization.hpp>

#include "parameter.h"

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    /// Parameter structure
    struct HPX_EXPORT Parameter : ::Par
    {
    private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive &ar, const unsigned int version) 
        {
            ar & lambda;
            ar & allowedl;
            ar & loglevel;
            ar & output;
            ar & output_stdout;
            ar & stencilsize;
            ar & linearbounds;
            ar & coarsestencilsize;
            ar & integrator;
            ar & nt0;
            ar & nx0;
            ar & minx0;
            ar & maxx0;
            ar & dx0;
            ar & dt0;
        }
    };

///////////////////////////////////////////////////////////////////////////////
}}}

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the id_type serialization format
#include <hpx/config/warnings_suffix.hpp>

#endif 
