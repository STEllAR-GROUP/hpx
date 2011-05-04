//  Copyright (c) 2009 Matt Anderson
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_PARAMETER_OCT_19_2009_0834AM)
#define HPX_COMPONENTS_PARAMETER_OCT_19_2009_0834AM

#include <boost/cstdint.hpp>
#include <boost/serialization/serialization.hpp>

#include "parameter.h"

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr 
{
    /// Parameter structure
    struct HPX_COMPONENT_EXPORT Parameter_impl : ::Par
    {
    private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive &ar, const unsigned int version);
    };

    struct HPX_COMPONENT_EXPORT Parameter  {
      boost::shared_ptr< Parameter_impl > p;
      Parameter() : p(new Parameter_impl) {
      }
      public:
        Parameter_impl * operator -> () {
          return p.get();
        }

        Parameter_impl const * operator -> () const {
          return p.get();
        }

      private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive &ar, const unsigned int version);
    };

///////////////////////////////////////////////////////////////////////////////
}}}

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the id_type serialization format
#include <hpx/config/warnings_suffix.hpp>

#endif 
