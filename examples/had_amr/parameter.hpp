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

class Array3D {
    size_t m_width, m_height;
    std::vector<int> m_data;
  public:
    Array3D(size_t x, size_t y, size_t z, int init = 0):
         m_width(x), m_height(y), m_data(x*y*z, init)
      {}
    int& operator()(size_t x, size_t y, size_t z) {
    return m_data.at(x + y * m_width + z * m_width * m_height);
  }
};

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
            ar & energy;
        }
    };

///////////////////////////////////////////////////////////////////////////////
}}}

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the id_type serialization format
#include <hpx/config/warnings_suffix.hpp>

#endif 
