//  Copyright (c) 2015-2017 John Biddiscombe
//  Copyright (c) 2017      Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/parcelport_libfabric/config/defines.hpp>

#include <rdma/fi_eq.h>

namespace hpx { namespace parcelset { namespace policies { namespace libfabric {
    struct parcelport;

    struct rma_base
    {
        rma_base() = default;
        virtual ~rma_base() = default;

        //  A placeholder to allow sender or rma_received subclasses to gracefully
        // handle an error on the network
        virtual void handle_error(struct fi_cq_err_entry err) = 0;
    };
}}}}    // namespace hpx::parcelset::policies::libfabric
