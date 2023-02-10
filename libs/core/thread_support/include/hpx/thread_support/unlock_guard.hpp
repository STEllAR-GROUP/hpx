//  Copyright (c) 2007-2008 Chirag Dekate, Hartmut Kaiser
//  Copyright (c) 2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file unlock_guard.hpp

#pragma once

#include <hpx/config.hpp>

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // This is a helper structure to make sure a lock gets unlocked and locked
    // again in a scope.

    /// The class \c unlock_guard is a mutex wrapper that provides a convenient
    /// mechanism for releasing a mutex for the duration of a scoped block.
    /// \details \c unlock_guard performs the opposite functionality of
    ///          \c lock_guard.
    ///          When a \c lock_guard object is created, it attempts to take
    ///          ownership of the mutex it is given. When control leaves the
    ///          scope in which the \c lock_guard object was created, the
    ///          \c lock_guard is destructed and the mutex is released.
    ///          Accordingly, when an \c unlock_guard object is created, it
    ///          attempts to release the ownership of the mutex it is given. So,
    ///          when control leaves the scope in which the \c unlock_guard
    ///          object was created, the \c unlock_guard is destructed and the
    ///          mutex is owned again. In this way, the mutex is unlocked in the
    ///          constructor and locked in the destructor, so that one can have
    ///          an unlocked section within a locked one.
    template <typename Mutex>
    class unlock_guard
    {
    public:
        using mutex_type = Mutex;

        explicit constexpr unlock_guard(Mutex& m) noexcept
          : m_(m)
        {
            m_.unlock();
        }

        HPX_NON_COPYABLE(unlock_guard);

        ~unlock_guard()
        {
            m_.lock();
        }

    private:
        Mutex& m_;
    };
}    // namespace hpx

namespace hpx::util {

    template <typename Mutex>
    using unlock_guard HPX_DEPRECATED_V(1, 9,
        "hpx::util::unlock_guard is deprecated, use hpx::unlock_guard "
        "instead") = hpx::unlock_guard<Mutex>;
}    // namespace hpx::util
