//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include "serialize_mpreal.hpp"

namespace boost { namespace serialization 
{
    ///////////////////////////////////////////////////////////////////////////
    template<class Archive>
    void load(Archive& ar, mpfr::mpreal& d, unsigned int version)
    {
        std::string s;
        ar & s;
        d = s.c_str();
    }

    ///////////////////////////////////////////////////////////////////////////
    template<class Archive>
    void save(Archive& ar, mpfr::mpreal const& d, unsigned int version)
    {
        std::string s(d.to_string());
        ar & s;
    }

    ///////////////////////////////////////////////////////////////////////////
    // explicit instantiation for the correct archive types
#if HPX_USE_PORTABLE_ARCHIVES != 0
    template HPX_COMPONENT_EXPORT 
    void save(hpx::util::portable_binary_oarchive&, mpfr::mpreal const& d, unsigned int version);

    template HPX_COMPONENT_EXPORT 
    void load(hpx::util::portable_binary_iarchive&, mpfr::mpreal& d, unsigned int version);
#else
    template HPX_COMPONENT_EXPORT 
    void save(boost::archive::binary_oarchive&, mpfr::mpreal const& d, unsigned int version);

    template HPX_COMPONENT_EXPORT 
    void load(boost::archive::binary_iarchive&, mpfr::mpreal& d, unsigned int version);
#endif
}}

