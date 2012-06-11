//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/asio/io_service.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    // generate unique parcel id
    naming::gid_type parcel::generate_unique_id()
    {
        static boost::atomic<boost::uint64_t> id(0);

        error_code ec;        // ignore all errors
        boost::uint32_t locality_id = hpx::get_locality_id(ec);
        naming::gid_type result = naming::get_gid_from_locality_id(locality_id);
        result.set_lsb(++id);
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive>
    void parcel::save(Archive & ar, const unsigned int version) const
    {
        ar << parcel_id_;
        ar << destination_id_;    // don't increment global refcnt
        ar << destination_addr_;

        // If we have a source id, serialize it.
        bool has_source_id = source_id_;
        ar << has_source_id;
        if (has_source_id)
            ar << source_id_;
        ar << action_;

        // If we have a continuation, serialize it.
        bool has_continuations = continuation_;
        ar << has_continuations;
        if (has_continuations)
            ar << continuation_;

        ar << start_time_;
        ar << creation_time_;
    }

    template <typename Archive>
    void parcel::load(Archive & ar, const unsigned int version)
    {
        if (version > HPX_PARCEL_VERSION)
        {
            HPX_THROW_EXCEPTION(version_too_new,
                "parcel::load",
                "trying to load parcel with unknown version");
        }

        bool has_continuation = false;
        bool has_source_id = false;

        ar >> parcel_id_;
        ar >> destination_id_;
        ar >> destination_addr_;

        // Check for a source id.
        ar >> has_source_id;
        if (has_source_id)
            ar >> source_id_;

        ar >> action_;

        // Check for a continuation.
        ar >> has_continuation;
        if (has_continuation)
            ar >> continuation_;

        ar >> start_time_;
        ar >> creation_time_;
    }

    ///////////////////////////////////////////////////////////////////////////
    // explicit instantiation for the correct archive types
    template HPX_EXPORT void
    parcel::save(util::portable_binary_oarchive&, const unsigned int version) const;

    template HPX_EXPORT void
    parcel::load(util::portable_binary_iarchive&, const unsigned int version);

    ///////////////////////////////////////////////////////////////////////////
    std::ostream& operator<< (std::ostream& os, parcel const& p)
    {
        os << "(" << p.destination_id_
           << ":" << p.destination_addr_
           << ":" << p.action_->get_action_name() << ")";
        return os;
    }
}}

