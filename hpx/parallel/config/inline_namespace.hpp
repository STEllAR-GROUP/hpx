//  Copyright (c) 2014-2017 Hartmut Kaiser
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
    HPX_INLINE_NAMESPACE(v3) {}
    HPX_INLINE_NAMESPACE(concurrency_v2) {}

#if !defined(HPX_HAVE_CXX11_INLINE_NAMESPACES)
    using namespace v1;
    using namespace v2;
    using namespace v3;
    using namespace concurrency_v2;
#endif

    namespace execution
    {
        HPX_INLINE_NAMESPACE(v1) {}

#if !defined(HPX_HAVE_CXX11_INLINE_NAMESPACES)
        using namespace v1;
#endif
    }
}}

#endif
