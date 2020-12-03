//  Copyright (c)      2014 Thomas Heller
//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/runtime_distributed.hpp>
#include <hpx/runtime/parcelset/locality.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/runtime_fwd.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/serialization/string.hpp>
#include <hpx/type_support/unused.hpp>
#include <hpx/util/ios_flags_saver.hpp>

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
        HPX_UNUSED(ar);
#endif
    }

    void locality::load(serialization::input_archive& ar, const unsigned int)
    {
#if defined(HPX_HAVE_NETWORKING)
        std::string t;
        ar >> t;
        if(t.empty()) return;
        HPX_ASSERT(get_runtime_ptr());
        impl_ = get_runtime_distributed()
                    .get_parcel_handler()
                    .create_locality(t)
                    .impl_;
        impl_->load(ar);
        HPX_ASSERT(impl_->valid());
#else
        HPX_THROW_EXCEPTION(invalid_status,
            "locality::load",
            "this shouldn't be called if networking is disabled");
        HPX_UNUSED(ar);
#endif
    }

    std::ostream& operator<< (std::ostream& os, endpoints_type const& endpoints)
    {
        hpx::util::ios_flags_saver ifs(os);
        os << "[ ";
        for (endpoints_type::value_type const& loc : endpoints)
        {
            os << "(" << loc.first << ":" << loc.second << ") ";
        }
        os << "]";

        return os;
    }
}}
