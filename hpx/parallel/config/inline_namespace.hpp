//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_CONFIG_INLINE_NAMESPACE_2014_JUL_03_0759PM)
#define HPX_PARALLEL_CONFIG_INLINE_NAMESPACE_2014_JUL_03_0759PM

#include <hpx/config/inline_namespace.hpp>

namespace hpx { namespace parallel
{
    HPX_INLINE_NAMESPACE(v1) {}
    HPX_INLINE_NAMESPACE(v2) {}

#if defined(BOOST_NO_CXX11_INLINE_NAMESPACES)
    using namespace v1;
    using namespace v2;
#endif
}}

#endif
