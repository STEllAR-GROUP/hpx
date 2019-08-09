//  Copyright (c) 2019 Mikael Simberg
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/concurrency/thread_name.hpp>

#include <string>

namespace hpx { namespace detail {
    std::string& thread_name()
    {
        static HPX_NATIVE_TLS std::string thread_name_;
        return thread_name_;
    }
}}    // namespace hpx::detail
