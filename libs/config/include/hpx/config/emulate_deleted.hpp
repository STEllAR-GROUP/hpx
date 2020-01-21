//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CONFIG_EMULATE_DELETED_JAN_06_2013_0919PM
#define HPX_CONFIG_EMULATE_DELETED_JAN_06_2013_0919PM

/// Marks a class as non-copyable and non-movable.
#define HPX_NON_COPYABLE(cls)                                                  \
    cls(cls const&) = delete;                                                  \
    cls(cls&&) = delete;                                                       \
    cls& operator=(cls const&) = delete;                                       \
    cls& operator=(cls&&) = delete /**/

#endif
