//  Copyright (c) 2019 The STE||AR GROUP
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_ASSERTION_FORCE_LINKING_HPP)
#define HPX_ASSERTION_FORCE_LINKING_HPP

#include <hpx/config.hpp>
#include <hpx/assertion/evaluate_assert.hpp>
#include <hpx/assertion/source_location.hpp>

#include <string>

namespace hpx { namespace assertion
{
    struct force_linking_helper
    {
#if defined(HPX_DEBUG)
        void (*handle_assert)(
            source_location const&, const char*, std::string const&);
#endif
    };

    force_linking_helper& force_linking();
}}

#endif
