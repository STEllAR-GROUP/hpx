//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/threads/thread_data_fwd.hpp

#if !defined(HPX_THREADS_THREAD_DATA_FWD_AUG_11_2015_0228PM)
#define HPX_THREADS_THREAD_DATA_FWD_AUG_11_2015_0228PM

#include <hpx/config.hpp>
#include <hpx/coroutines/coroutine_fwd.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/coroutines/thread_id_type.hpp>
#include <hpx/errors.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/functional/unique_function.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace hpx {
    /// \cond NOINTERNAL
    class HPX_EXPORT thread;
    /// \endcond
}    // namespace hpx

namespace hpx { namespace threads {
    /// \cond NOINTERNAL
    struct HPX_EXPORT topology;

    class HPX_EXPORT executor;
    /// \endcond
}}    // namespace hpx::threads

#endif
