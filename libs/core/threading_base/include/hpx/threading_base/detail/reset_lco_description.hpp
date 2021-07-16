//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_DESCRIPTION)

#include <hpx/modules/errors.hpp>
#include <hpx/threading_base/thread_description.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>

namespace hpx { namespace threads { namespace detail {

    struct reset_lco_description
    {
        reset_lco_description(threads::thread_id_type const& id,
            util::thread_description const& description,
            error_code& ec = throws)
          : id_(id)
          , ec_(ec)
        {
            old_desc_ =
                threads::set_thread_lco_description(id_, description, ec_);
        }

        ~reset_lco_description()
        {
            threads::set_thread_lco_description(id_, old_desc_, ec_);
        }

        threads::thread_id_type id_;
        util::thread_description old_desc_;
        error_code& ec_;
    };
}}}    // namespace hpx::threads::detail

#endif    // HPX_HAVE_THREAD_DESCRIPTION
