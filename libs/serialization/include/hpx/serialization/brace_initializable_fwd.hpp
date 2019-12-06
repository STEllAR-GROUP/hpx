//  Copyright (c) 2019 Jan Melech
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_BRACE_INITIALIZABLE_FWD_HPP
#define HPX_SERIALIZATION_BRACE_INITIALIZABLE_FWD_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX17_STRUCTURED_BINDINGS) &&                             \
    defined(HPX_HAVE_CXX17_IF_CONSTEXPR)
namespace hpx { namespace serialization {

    template <typename Archive, typename T>
    void serialize_struct(Archive& ar, T& t, const unsigned int);
}}    // namespace hpx::serialization
#endif

#endif
