//  Copyright (c)      2014 Thomas Heller
//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/serialization/string.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    void locality::save(serialization::output_archive& ar, const unsigned int) const
    {
        std::string t = type();
        ar << t;
        if(t.empty()) return;
        impl_->save(ar);
    }

    void locality::load(serialization::input_archive& ar, const unsigned int)
    {
        std::string t;
        ar >> t;
        if(t.empty()) return;
        HPX_ASSERT(get_runtime_ptr());
        impl_ = get_runtime().get_parcel_handler().create_locality(t).impl_;
        impl_->load(ar);
        HPX_ASSERT(impl_->valid());
    }

    std::ostream& operator<< (std::ostream& os, endpoints_type const& endpoints)
    {
        boost::io::ios_flags_saver ifs(os);
        os << "[ ";
        for (endpoints_type::value_type const& loc : endpoints)
        {
            os << "(" << loc.first << ":" << loc.second << ") ";
        }
        os << "]";

        return os;
    }
}}
