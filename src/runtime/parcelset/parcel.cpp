//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/serialization/unique_ptr.hpp>

#include <boost/atomic.hpp>

#include <cstdint>
#include <sstream>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    ///////////////////////////////////////////////////////////////////////////
    // generate unique parcel id
    naming::gid_type parcel::generate_unique_id(
        std::uint32_t locality_id_default)
    {
        static boost::atomic<std::uint64_t> id(0);

        error_code ec(lightweight);        // ignore all errors
        std::uint32_t locality_id = hpx::get_locality_id(ec);
        if (locality_id == naming::invalid_locality_id)
            locality_id = locality_id_default;

        naming::gid_type result = naming::get_gid_from_locality_id(locality_id);
        result.set_lsb(++id);
        return result;
    }

    void parcel::serialize(serialization::input_archive & ar, unsigned)
    {
        ar & data_;
        if(data_.has_source_id_)
        {
            ar & source_id_;
        }
        else
        {
            source_id_ = naming::invalid_id;
        }
        ar & dests_;
        ar & addrs_;
        ar & cont_;
        ar & action_;
    }

    void parcel::serialize(serialization::output_archive & ar, unsigned)
    {
        ar & data_;
        if(data_.has_source_id_)
        {
            ar & source_id_;
        }
        ar & dests_;
        ar & addrs_;
        ar & cont_;
        ar & action_;
    }

    ///////////////////////////////////////////////////////////////////////////
    std::ostream& operator<< (std::ostream& os, parcel const& p)
    {
#if defined(HPX_SUPPORT_MULTIPLE_PARCEL_DESTINATIONS)
        os << "(";
        if (!p.dests_.empty())
            os << p.dests_[0] << ":" << p.addrs_[0] << ":";
#else
        os << "(" << p.dests_ << ":" << p.addrs_ << ":";
#endif
        os << p.action_->get_action_name() << ")";

        return os;
    }

    std::string dump_parcel(parcel const& p)
    {
        std::ostringstream os;
        os << p;
        return os.str();
    }
}}

