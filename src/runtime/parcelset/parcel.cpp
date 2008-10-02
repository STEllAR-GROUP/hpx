//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>

#include <boost/asio.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>

///////////////////////////////////////////////////////////////////////////////
// this is the current version of the parcel serialization format
// this definition needs to be in the global namespace
BOOST_CLASS_VERSION(hpx::parcelset::parcel, HPX_PARCEL_VERSION)
BOOST_CLASS_TRACKING(hpx::parcelset::parcel, boost::serialization::track_never)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    template<class Archive>
    void parcel::save(Archive & ar, const unsigned int version) const
    {
        ar << tag_;
        ar << destination_id_;
        ar << destination_addr_;
        ar << source_id_;
        ar << action_;
        bool has_continuations = continuation_ && !continuation_->empty();
        ar << has_continuations;
        if (has_continuations)
            ar << continuation_;
        ar << start_time_;
        ar << creation_time_;
    }

    template<class Archive>
    void parcel::load(Archive & ar, const unsigned int version)
    {
        if (version > HPX_PARCEL_VERSION) {
            throw exception(version_too_new, 
                "trying to load parcel with unknown version");
        }

        bool has_continuation = false;
        ar >> tag_;
        ar >> destination_id_;
        ar >> destination_addr_;
        ar >> source_id_;
        ar >> action_;
        ar >> has_continuation;
        if (has_continuation)
            ar >> continuation_;
        ar >> start_time_;
        ar >> creation_time_;
    }

    // explicit instantiation for the correct archive types
    template HPX_EXPORT void 
    parcel::save(util::portable_binary_oarchive&, const unsigned int version) const;

    template HPX_EXPORT void 
    parcel::load(util::portable_binary_iarchive&, const unsigned int version);
    
///////////////////////////////////////////////////////////////////////////////
}}
