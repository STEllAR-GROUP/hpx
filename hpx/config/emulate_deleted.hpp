//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_CONFIG_EMULATE_DELETED_JAN_06_2013_0919PM)
#define HPX_CONFIG_EMULATE_DELETED_JAN_06_2013_0919PM

#include <hpx/config.hpp>

#define HPX_DELETE_COPY_CTOR(cls)                                             \
    cls(cls const&) = delete                                                  \
/**/

#define HPX_DELETE_COPY_ASSIGN(cls)                                           \
    cls& operator=(cls const&) = delete                                       \
/**/

#define HPX_DELETE_MOVE_CTOR(cls)                                             \
    cls(cls&&) = delete                                                       \
/**/

#define HPX_DELETE_MOVE_ASSIGN(cls)                                           \
    cls& operator=(cls&&) = delete                                            \
/**/

#define HPX_NON_COPYABLE(cls)                                                 \
    HPX_DELETE_COPY_CTOR(cls);                                                \
    HPX_DELETE_COPY_ASSIGN(cls);                                              \
    HPX_DELETE_MOVE_CTOR(cls);                                                \
    HPX_DELETE_MOVE_ASSIGN(cls)                                               \
/**/

#define HPX_MOVABLE_ONLY(cls)                                                 \
    HPX_DELETE_COPY_CTOR(cls);                                                \
    HPX_DELETE_COPY_ASSIGN(cls)                                               \
/**/

#endif
