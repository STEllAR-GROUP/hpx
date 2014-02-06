//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_ENABLE_DISABLE_HPP_FEB_05_2014_1244PM)
#define HPX_PARCELSET_ENABLE_DISABLE_HPP_FEB_05_2014_1244PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>

namespace hpx { namespace parcelset
{
    struct disable
    {
        disable()
          : old_state_(hpx::get_runtime().get_parcel_handler().enable(false))
        {}
        ~disable()
        {
            hpx::get_runtime().get_parcel_handler().enable(old_state_);
        }

        bool old_state_;
    };

    struct enable
    {
        enable(disable& d)
          : old_state_(hpx::get_runtime().get_parcel_handler().enable(d.old_state_))
        {}
        ~enable()
        {
            hpx::get_runtime().get_parcel_handler().enable(old_state_);
        }

        bool old_state_;
    };
}}

#endif
