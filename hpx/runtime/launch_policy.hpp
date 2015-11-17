//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/launch_policy.hpp

#if !defined(HPX_RUNTIME_LAUNCH_POLICY_AUG_13_2015_0647PM)
#define HPX_RUNTIME_LAUNCH_POLICY_AUG_13_2015_0647PM

#include <hpx/config.hpp>

#include <boost/detail/scoped_enum_emulation.hpp>

#include <type_traits>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// Launch policy for \a hpx::async
    BOOST_SCOPED_ENUM_START(launch)
    {
        async = 0x01,
        deferred = 0x02,
        task = 0x04,        // see N3632
        sync = 0x08,
        fork = 0x10,        // same as async, but forces continuation stealing

        sync_policies = 0x0a,       // sync | deferred
        async_policies = 0x15,      // async | task | fork
        all = 0x1f                  // async | deferred | task | sync | fork
    };
    BOOST_SCOPED_ENUM_END

    /// \cond NOINTERNAL
    inline bool
    operator&(BOOST_SCOPED_ENUM(launch) lhs, BOOST_SCOPED_ENUM(launch) rhs)
    {
        return (static_cast<int>(lhs) & static_cast<int>(rhs)) != 0;
    }

    inline BOOST_SCOPED_ENUM(launch)
    operator|(BOOST_SCOPED_ENUM(launch) lhs, BOOST_SCOPED_ENUM(launch) rhs)
    {
        return static_cast<BOOST_SCOPED_ENUM(launch)>(
            static_cast<int>(lhs) | static_cast<int>(rhs));
    }
    /// \endcond
}

#endif
