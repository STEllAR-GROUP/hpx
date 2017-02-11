//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file get_thread_name.hpp

#if !defined(HPX_RUNTIME_GET_THREAD_NAME_HPP)
#define HPX_RUNTIME_GET_THREAD_NAME_HPP

#include <hpx/config.hpp>
#include <hpx/util/itt_notify.hpp>

#include <string>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    /// \brief Return the name of the calling thread.
    ///
    /// This function returns the name of the calling thread. This name uniquely
    /// identifies the thread in the context of HPX. If the function is called
    /// while no HPX runtime system is active, the result will be "<unknown>".
    HPX_API_EXPORT std::string get_thread_name();

    /// \cond NOINTERNAL
    HPX_API_EXPORT hpx::util::itt::domain const& get_thread_itt_domain();
    /// \endcond
}

#endif
