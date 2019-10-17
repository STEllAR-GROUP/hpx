////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#ifndef HPX_EXECUTION_RESOURCE_BASE_HPP
#define HPX_EXECUTION_RESOURCE_BASE_HPP

namespace hpx { namespace basic_execution {

    /// TODO: implement, this is currently just a dummy
    struct resource_base
    {
        virtual ~resource_base() = default;
    };
}}    // namespace hpx::basic_execution

#endif
