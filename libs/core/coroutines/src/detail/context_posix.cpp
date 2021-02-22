//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if !defined(HPX_HAVE_GENERIC_CONTEXT_COROUTINES) &&                           \
    !(defined(__linux) || defined(linux) || defined(__linux__)) &&             \
    (defined(_POSIX_VERSION) || defined(__bgq__) || defined(__powerpc__) ||    \
        defined(__s390x__))
#include <hpx/coroutines/detail/context_posix.hpp>

#include <cstddef>

namespace hpx { namespace threads { namespace coroutines { namespace detail {
    namespace posix {

        std::ptrdiff_t ucontext_context_impl_base::default_stack_size =
            SIGSTKSZ;
}}}}}    // namespace hpx::threads::coroutines::detail::posix

#endif
