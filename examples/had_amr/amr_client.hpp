//  Copyright (c) 2007-2009 Hartmut Kaiser
//  Copyright (c) 2007 Richard D. Guidry Jr.
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#if !defined(HPX_COMPONENTS_AMR_CLIENT_OCT_19_2009_0834AM)
#define HPX_COMPONENTS_AMR_CLIENT_OCT_19_2009_0834AM

#include <boost/cstdint.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/serialization.hpp>

#include "amr_client.h"
#include <hpx/util/safe_bool.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
// Version of id_type

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server
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
            ar & stencilsize;
        }
    };

///////////////////////////////////////////////////////////////////////////////
}}}}

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the id_type serialization format
#include <hpx/config/warnings_suffix.hpp>

#endif 
