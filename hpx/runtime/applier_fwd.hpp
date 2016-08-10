
//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file applier_fwd.hpp

#ifndef HPX_RUNTIME_APPLIER_FWD_HPP
#define HPX_RUNTIME_APPLIER_FWD_HPP

#include <hpx/config.hpp>

namespace hpx
{
    /// \namespace applier
    ///
    /// The namespace \a applier contains all definitions needed for the
    /// class \a hpx#applier#applier and its related functionality. This
    /// namespace is part of the HPX core module.
    namespace applier
    {
        class HPX_API_EXPORT applier;

        /// The function \a get_applier returns a reference to the (thread
        /// specific) applier instance.
        HPX_API_EXPORT applier& get_applier();

        /// The function \a get_applier returns a pointer to the (thread
        /// specific) applier instance. The returned pointer is NULL if the
        /// current thread is not known to HPX or if the runtime system is not
        /// active.
        HPX_API_EXPORT applier* get_applier_ptr();
    }
}

#endif
