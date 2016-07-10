//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_REPORT_ERROR_HPP)
#define HPX_RUNTIME_REPORT_ERROR_HPP

#include <hpx/config.hpp>

#include <boost/exception_ptr.hpp>

namespace hpx
{
    HPX_API_EXPORT void report_error(std::size_t num_thread,
        boost::exception_ptr const& e);

    HPX_API_EXPORT void report_error(boost::exception_ptr const& e);
}

#endif

