//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/asio.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include <hpx/parcelset/parcel.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    template<class Archive>
    void parcel::save(Archive & ar, const unsigned int version) const
    {
        ar << tag_;
        ar << destination_id_.id_;
        ar << destination_addr_;
        ar << source_id_.id_;
        ar << action_;
        ar << cont_;
    }

    template<class Archive>
    void parcel::load(Archive & ar, const unsigned int version)
    {
        if (version > HPX_PARCEL_VERSION) {
            throw exception(version_too_new, 
                "trying to load parcel with unknown version");
        }

        ar >> tag_;
        ar >> destination_id_.id_;
        ar >> destination_addr_;
        ar >> source_id_.id_;
        ar >> action_;
        ar >> cont_;
    }

    // explicit instantiation for the correct archive types
    template void 
    parcel::save(util::portable_binary_oarchive&, const unsigned int version) const;

    template void 
    parcel::load(util::portable_binary_iarchive&, const unsigned int version);
    
///////////////////////////////////////////////////////////////////////////////
}}
