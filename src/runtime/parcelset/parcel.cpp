//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/serialize_intrusive_ptr.hpp>

#include <boost/serialization/string.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    // generate unique parcel id
    naming::gid_type parcel::generate_unique_id()
    {
        static boost::atomic<boost::uint64_t> id(0);

        error_code ec(lightweight);        // ignore all errors
        boost::uint32_t locality_id = hpx::get_locality_id(ec);
        naming::gid_type result = naming::get_gid_from_locality_id(locality_id);
        result.set_lsb(++id);
        return result;
    }

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////////
        template <typename Archive>
        void parcel_data::save(Archive & ar, const unsigned int version) const
        {
            ar << parcel_id_;

            // Serialize only one destination, if needed.
            bool has_one_dest = (gids_.size() == 1) ? true : false;
            ar << has_one_dest;
            if (has_one_dest)
                ar << gids_[0] << addrs_[0];
            else
                ar << gids_ << addrs_;

            // If we have a source id, serialize it.
            bool has_source_id = source_id_;
            ar << has_source_id;
            if (has_source_id)
                ar << source_id_;

            ar << action_;

            // If we have a continuation, serialize it.
            bool has_continuations = static_cast<bool>(continuation_);
            ar << has_continuations;
            if (has_continuations) {
                actions::continuation const* c = continuation_.get();
                ar << c;
            }

            ar << start_time_;
            ar << creation_time_;
        }

        template <typename Archive>
        void parcel_data::load(Archive & ar, const unsigned int version)
        {
            if (version > HPX_PARCEL_VERSION)
            {
                HPX_THROW_EXCEPTION(version_too_new,
                    "parcel::load",
                    "trying to load parcel with unknown version");
            }

            bool has_continuation = false;
            bool has_source_id = false;
            bool has_one_dest = false;

            ar >> parcel_id_;

            // Properly de-serialize destinations.
            ar >> has_one_dest;
            if (has_one_dest) {
                gids_.resize(1);
                addrs_.resize(1);
                ar >> gids_[0] >> addrs_[0];
            }
            else {
                ar >> gids_ >> addrs_;
            }

            // Check for a source id.
            ar >> has_source_id;
            if (has_source_id)
                ar >> source_id_;

            ar >> action_;

            // Check for a continuation.
            ar >> has_continuation;
            if (has_continuation) {
                actions::continuation* c = 0;
                ar >> c;
                continuation_.reset(c);
            }

            ar >> start_time_;
            ar >> creation_time_;
        }

        ///////////////////////////////////////////////////////////////////////
        // explicit instantiation for the correct archive types
        template HPX_EXPORT void
        parcel_data::save(util::portable_binary_oarchive&, const unsigned int) const;

        template HPX_EXPORT void
        parcel_data::load(util::portable_binary_iarchive&, const unsigned int);

        ///////////////////////////////////////////////////////////////////////////
        std::ostream& operator<< (std::ostream& os, parcel_data const& p)
        {
            os << "(";
            if (!p.gids_.empty())
                os << p.gids_[0] << ":" << p.addrs_[0] << ":";

            os << p.action_->get_action_name() << ")";
            return os;
        }
    }

    template <typename Archive>
    void parcel::serialize(Archive & ar, const unsigned int version)
    {
        ar & data_;
    }

    ///////////////////////////////////////////////////////////////////////////
    // explicit instantiation for the correct archive types
    template HPX_EXPORT void
    parcel::serialize(util::portable_binary_oarchive&, const unsigned int);

    template HPX_EXPORT void
    parcel::serialize(util::portable_binary_iarchive&, const unsigned int);

    std::ostream& operator<< (std::ostream& os, parcel const& p)
    {
        BOOST_ASSERT(p.data_);
        os << *p.data_;
        return os;
    }
}}

namespace hpx { namespace traits
{
    std::size_t 
    type_size<hpx::parcelset::parcel>::call(hpx::parcelset::parcel const& parcel_)
    {
        return sizeof(hpx::parcelset::parcel) + parcel_.get_action()->get_type_size();
    }
}}

