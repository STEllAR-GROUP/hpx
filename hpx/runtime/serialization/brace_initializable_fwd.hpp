//  Copyright (c) 2019 Jan Melech
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_BRACE_INITIALIZABLE_FWD_HPP
#define HPX_SERIALIZATION_BRACE_INITIALIZABLE_FWD_HPP

#include <hpx/config/automatic_struct_serialization.hpp>

namespace hpx { namespace serialization
{
#if defined(HPX_SUPPORT_AUTOMATIC_STRUCT_SERIALIZATION)
    template <typename Archive, typename T>
    void serialize_struct(Archive& ar, T& t, const unsigned int);
#endif
}}

#endif
