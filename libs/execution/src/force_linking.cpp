//  Copyright (c) 2019-2020 The STE||AR GROUP
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/execution/executors/polymorphic_executor.hpp>
#include <hpx/execution/force_linking.hpp>

namespace hpx { namespace execution {

    // reference all symbols that have to be explicitly linked with the core
    // library
    force_linking_helper& force_linking()
    {
        static force_linking_helper helper{
            &parallel::execution::detail::throw_bad_polymorphic_executor};
        return helper;
    }
}}    // namespace hpx::execution
