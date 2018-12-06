//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/util/detail/empty_function.hpp>

#include <hpx/throw_exception.hpp>

namespace hpx { namespace util { namespace detail
{
    HPX_NORETURN void throw_bad_function_call()
    {
        hpx::throw_exception(bad_function_call,
            "empty function object should not be used",
            "empty_function::operator()");
    }
}}}
