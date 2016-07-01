//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file runtime_mode.hpp

#ifndef HPX_RUNTIME_RUNTIME_MODE_HPP
#define HPX_RUNTIME_RUNTIME_MODE_HPP

#include <hpx/config.hpp>

#include <string>

namespace hpx
{
    /// A HPX runtime can be executed in two different modes: console mode
    /// and worker mode.
    enum runtime_mode
    {
        runtime_mode_invalid = -1,
        runtime_mode_console = 0,   ///< The runtime is the console locality
        runtime_mode_worker = 1,    ///< The runtime is a worker locality
        runtime_mode_connect = 2,   ///< The runtime is a worker locality
                                    ///< connecting late
        runtime_mode_default = 3,   ///< The runtime mode will be determined
                                    ///< based on the command line arguments
        runtime_mode_last
    };

    /// Get the readable string representing the name of the given runtime_mode
    /// constant.
    HPX_API_EXPORT char const* get_runtime_mode_name(runtime_mode state);

    /// Get the internal representation (runtime_mode constant) from the 
    /// readable string representing the name.
    HPX_API_EXPORT runtime_mode get_runtime_mode_from_name(std::string const& mode);
}

#endif /*HPX_RUNTIME_RUNTIME_MODE_HPP*/
