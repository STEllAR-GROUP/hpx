//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/modules/functional.hpp>

#include <hpx/parcelset_base/parcelset_base_fwd.hpp>

#include <system_error>

namespace hpx::parcelset::policies {

    struct message_handler
    {
        enum flush_mode
        {
            flush_mode_timer = 0,
            flush_mode_background_work = 1,
            flush_mode_buffer_full = 2
        };

        using write_handler_type = hpx::function<void(
            std::error_code const&, parcelset::parcel const&)>;

        virtual ~message_handler() = default;

        virtual void put_parcel(parcelset::locality const& dest, parcel p,
            write_handler_type f) = 0;
        virtual bool flush(flush_mode mode, bool stop_buffering = false) = 0;
    };
}    // namespace hpx::parcelset::policies

#endif
