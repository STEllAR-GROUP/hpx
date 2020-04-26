//  Copyright (c) 2019-2020 STE||AR Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_FORCE_LINKING_HPP
#define HPX_SERIALIZATION_FORCE_LINKING_HPP

namespace hpx { namespace serialization {
    struct force_linking_helper
    {
        void (*dummy)();
    };

    force_linking_helper& force_linking();
}}    // namespace hpx::serialization

#endif
