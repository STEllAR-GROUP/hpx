//  Copyright (c) 2009 Matt Anderson
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include "../parameter.hpp"

#include <boost/serialization/string.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace nbody 
{
    template<class Archive>
    void Parameter_impl::serialize(Archive &ar, const unsigned int version) 
    {
        ar & output;
        ar & output_stdout;
        ar & loglevel;
        ar & rowsize;
        ar & input_file;
        ar & iList;
        ar & bilist;
        ar & dtime;
        ar & eps;
        ar & tolerance;
        ar & half_dt;
        ar & softening_2;
        ar & inv_tolerance_2;
        ar & iter;
        ar & num_bodies;
        ar & num_iterations;
        ar & part_mass;
        ar & bodies;
        ar & granularity;
        ar & num_pxpar;
        ar & extra_pxpar;
    }

    // explicit instantiation for the correct archive types
#if HPX_USE_PORTABLE_ARCHIVES != 0
    template HPX_COMPONENT_EXPORT void 
    Parameter_impl::serialize(util::portable_binary_iarchive&, 
        const unsigned int version);
    template HPX_COMPONENT_EXPORT void 
    Parameter_impl::serialize(util::portable_binary_oarchive&, 
        const unsigned int version);
#else
    template HPX_COMPONENT_EXPORT void 
    Parameter_impl::serialize(boost::archive::binary_iarchive&, 
        const unsigned int version);
    template HPX_COMPONENT_EXPORT void 
    Parameter_impl::serialize(boost::archive::binary_oarchive&, 
        const unsigned int version);
#endif

    ///////////////////////////////////////////////////////////////////////////
    template<class Archive>
    void Parameter::serialize(Archive &ar, const unsigned int version) 
    {
      ar & p;
    }

    // explicit instantiation for the correct archive types
#if HPX_USE_PORTABLE_ARCHIVES != 0
    template HPX_COMPONENT_EXPORT void 
    Parameter::serialize(util::portable_binary_iarchive&, 
        const unsigned int version);
    template HPX_COMPONENT_EXPORT void 
    Parameter::serialize(util::portable_binary_oarchive&, 
        const unsigned int version);
#else
    template HPX_COMPONENT_EXPORT void 
    Parameter::serialize(boost::archive::binary_iarchive&, 
        const unsigned int version);
    template HPX_COMPONENT_EXPORT void 
    Parameter::serialize(boost::archive::binary_oarchive&, 
        const unsigned int version);
#endif

}}}
