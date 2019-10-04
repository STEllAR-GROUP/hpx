//  Copyright (c)      2014 Thomas Heller
//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/errors.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/string.hpp>

#include <boost/io/ios_state.hpp>

#include <string>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset
{
    void locality::save(serialization::output_archive& ar, const unsigned int) const
    {
#if defined(HPX_HAVE_NETWORKING)
        std::string t = type();
        ar << t;
        if(t.empty()) return;
        impl_->save(ar);
#else
        HPX_THROW_EXCEPTION(invalid_status,
            "locality::save",
            "this shouldn't be called if networking is disabled");
#endif
    }

    void locality::load(serialization::input_archive& ar, const unsigned int)
    {
#if defined(HPX_HAVE_NETWORKING)
        std::string t;
        ar >> t;
        if(t.empty()) return;
        HPX_ASSERT(get_runtime_ptr());
        impl_ = get_runtime().get_parcel_handler().create_locality(t).impl_;
        impl_->load(ar);
        HPX_ASSERT(impl_->valid());
#else
        HPX_THROW_EXCEPTION(invalid_status,
            "locality::load",
            "this shouldn't be called if networking is disabled");
#endif
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
