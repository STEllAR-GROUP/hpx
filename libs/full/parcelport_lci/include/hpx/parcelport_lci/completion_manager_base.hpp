//  Copyright (c) 2023-2024 Jiakun Yan
//  Copyright (c) 2014-2023 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/modules/lci_base.hpp>

namespace hpx::parcelset::policies::lci {
    class HPX_EXPORT parcelport;
    struct completion_manager_base
    {
        completion_manager_base(parcelport* pp) noexcept
          : pp_(pp) {};
        virtual ~completion_manager_base() {}
        virtual LCI_comp_t alloc_completion() = 0;
        virtual void enqueue_completion(LCI_comp_t comp) = 0;
        virtual LCI_request_t poll() = 0;
        virtual LCI_comp_t get_completion_object()
        {
            return nullptr;
        }
        parcelport* pp_;
    };
}    // namespace hpx::parcelset::policies::lci

#endif
