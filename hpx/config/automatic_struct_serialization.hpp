//  Copyright (c) 2019 Jan Melech
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CONFIG_AUTOMATIC_STRUCT_SERIALIZATION_HPP
#define HPX_CONFIG_AUTOMATIC_STRUCT_SERIALIZATION_HPP

#include <hpx/config/defines.hpp>

#if defined(HPX_WITH_CXX17_STRUCTURED_BINDINGS) && defined (HPX_WITH_CXX17_IF_CONSTEXPR)
#define HPX_SUPPORT_AUTOMATIC_STRUCT_SERIALIZATION
#endif

#endif
