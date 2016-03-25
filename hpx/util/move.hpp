//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_MOVE_HPP
#define HPX_UTIL_MOVE_HPP

#include <hpx/config.hpp>

#if !defined(HPX_HAVE_CXX11_DELETED_FUNCTIONS)
#define HPX_MOVABLE_BUT_NOT_COPYABLE(TYPE)                                    \
    private:                                                                  \
        TYPE(TYPE const &);                                                   \
        TYPE& operator=(TYPE const &);                                        \
/**/
#else
#define HPX_MOVABLE_BUT_NOT_COPYABLE(TYPE)                                    \
    public:                                                                   \
        TYPE(TYPE const &) = delete;                                          \
        TYPE& operator=(TYPE const &) = delete;                               \
    private:                                                                  \
/**/
#endif

#endif
