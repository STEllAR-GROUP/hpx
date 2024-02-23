//  Copyright (c) 2020-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#ifdef HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION

#include <hpx/modules/debugging.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/threading_base/detail/reset_backtrace.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>

#include <memory>
#include <string>

namespace hpx::threads::detail {

    reset_backtrace::reset_backtrace(threads::thread_id_type id, error_code& ec)
      : id_(HPX_MOVE(id))
      , backtrace_(new hpx::util::backtrace())
#ifdef HPX_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
      , full_backtrace_(backtrace_->trace())
#endif
      , ec_(ec)
    {
#ifdef HPX_HAVE_THREAD_FULLBACKTRACE_ON_SUSPENSION
        threads::set_thread_backtrace(id_, full_backtrace_.c_str(), ec_);
#else
        threads::set_thread_backtrace(id_, backtrace_.get(), ec_);
#endif
    }

    reset_backtrace::~reset_backtrace()
    {
        threads::set_thread_backtrace(id_, nullptr, ec_);
    }
}    // namespace hpx::threads::detail

#endif    // HPX_HAVE_THREAD_BACKTRACE_ON_SUSPENSION
