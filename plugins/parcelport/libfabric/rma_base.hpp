//  Copyright (c) 2015-2017 John Biddiscombe
//  Copyright (c) 2017      Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_POLICIES_LIBFABRIC_RMA_BASE_HPP
#define HPX_PARCELSET_POLICIES_LIBFABRIC_RMA_BASE_HPP

#include <rdma/fi_eq.h>

namespace hpx {
namespace parcelset {
namespace policies {
namespace libfabric
{
    struct parcelport;

    enum rma_context_type {
        ctx_sender       = 0,
        ctx_receiver     = 1,
        ctx_rma_receiver = 2
    };

    // rma_base is base class for sender, receiver and rma_receiver. The first
    // N bytes must be occupied by an fi_context structure for correct libfabric
    // operation. rma_base cannot therefore have any virtual methods.
    // The first bytes of this object storage must contain the fi_context
    // structure needed by libfabric.
    // we provide an extra 'type' enum so that we can dispatch calls to the correct
    // object type when an error occurs (see controller poll_send_queue etc)
    struct rma_base
    {
        // check address of context is address of this object
        rma_base(rma_context_type ctx_type)
            : context_rma_type(ctx_type)
        {
            HPX_ASSERT(reinterpret_cast<void*>(&this->context_reserved_space)
                        == reinterpret_cast<void*>(&*this));
        }

        inline const rma_context_type& context_type() const { return context_rma_type; }

    private:
        // libfabric requires some space for it's internal bookkeeping
        fi_context       context_reserved_space;
        rma_context_type context_rma_type;

    };
}}}}

#endif
