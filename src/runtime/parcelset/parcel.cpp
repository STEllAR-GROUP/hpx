//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/serialization/unique_ptr.hpp>

#include <boost/atomic.hpp>

#include <sstream>
#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    void parcel::serialize(serialization::input_archive & ar, unsigned)
    {
        ar & data_;
        ar & cont_;
        ar & action_;
    }

    void parcel::serialize(serialization::output_archive & ar, unsigned)
    {
        ar & data_;
        ar & cont_;
        ar & action_;
    }

    ///////////////////////////////////////////////////////////////////////////
    std::ostream& operator<< (std::ostream& os, parcel const& p)
    {
        os << "(" << p.data_.dest_ << ":" << p.data_.addr_ << ":";
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

