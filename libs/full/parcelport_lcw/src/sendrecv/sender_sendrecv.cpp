//  Copyright (c) 2025 Jiakun Yan
//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//  Copyright (c)      2020 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCW)

#include <hpx/modules/lcw_base.hpp>
#include <hpx/parcelport_lcw/locality.hpp>
#include <hpx/parcelport_lcw/receiver_base.hpp>
#include <hpx/parcelport_lcw/sendrecv/sender_connection_sendrecv.hpp>
#include <hpx/parcelport_lcw/sendrecv/sender_sendrecv.hpp>

#include <memory>

namespace hpx::parcelset::policies::lcw {
    sender_sendrecv::connection_ptr sender_sendrecv::create_connection(
        int dest, parcelset::parcelport* pp)
    {
        return std::make_shared<sender_connection_sendrecv>(dest, pp);
    }
}    // namespace hpx::parcelset::policies::lcw

#endif
