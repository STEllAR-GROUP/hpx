//  Copyright (c) 2007-2010 Hartmut Kaiser, Matt Anderson
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_SERIALIZE_MPREAL_MAR_10_2010_1035AM)
#define HPX_COMPONENTS_SERIALIZE_MPREAL_MAR_10_2010_1035AM

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_free.hpp>

#include "mpreal.h"

namespace boost { namespace serialization 
{
    ///////////////////////////////////////////////////////////////////////////
    template<class Archive>
    void load(Archive& ar, mpfr::mpreal& d, unsigned int version);

    ///////////////////////////////////////////////////////////////////////////
    template<class Archive>
    void save(Archive& ar, mpfr::mpreal const& d, unsigned int version);
}}

BOOST_SERIALIZATION_SPLIT_FREE(mpfr::mpreal)

#endif
