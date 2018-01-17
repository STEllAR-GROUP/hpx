//  Copyright (c) 2018 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CONFIG_DECAY_T_HPP
#define HPX_CONFIG_DECAY_T_HPP

#include <hpx/config/defines.hpp>

#if defined(HPX_HAVE_CXX14_DECAY_T)
#define HPX_DECAY_T(T) std::decay_t<T>
#else
#define HPX_DECAY_T(T) typename std::decay<T>::type
#endif

#endif
