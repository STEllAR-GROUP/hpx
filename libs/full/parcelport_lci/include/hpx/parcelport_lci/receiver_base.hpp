//  Copyright (c) 2023-2024 Jiakun Yan
//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCELPORT_LCI)

#include <hpx/assert.hpp>
#include <hpx/parcelport_lci/header.hpp>
#include <hpx/parcelset/decode_parcels.hpp>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <deque>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

namespace hpx::parcelset::policies::lci {
    class HPX_EXPORT parcelport;
    struct buffer_wrapper
    {
        struct fake_allocator
        {
        };
        using allocator_type = fake_allocator;
        void* ptr;
        size_t length;
        buffer_wrapper() = default;
        buffer_wrapper(const buffer_wrapper& wrapper) = default;
        buffer_wrapper& operator=(const buffer_wrapper& wrapper) = default;
        explicit buffer_wrapper(const allocator_type& alloc)
        {
            HPX_UNUSED(alloc);
            ptr = nullptr;
            length = 0;
        }
        buffer_wrapper(
            const buffer_wrapper& wrapper, const allocator_type& alloc)
        {
            HPX_UNUSED(alloc);
            ptr = wrapper.ptr;
            length = wrapper.length;
        }
        ~buffer_wrapper() {}
        char& operator[](size_t i) const
        {
            HPX_ASSERT(i < length);
            char* p = (char*) ptr;
            return p[i];
        }
        void* data() const
        {
            return ptr;
        }
        size_t size() const
        {
            return length;
        }
        void allocate(size_t size)
        {
            ptr = new char[size];
            length = size;
        }
        void free()
        {
            HPX_ASSERT(ptr != nullptr);
            delete[](char*) ptr;
        }
    };

    struct request_wrapper_t
    {
        LCI_request_t request;
        request_wrapper_t()
        {
            request.flag = LCI_ERR_RETRY;
        }
        ~request_wrapper_t()
        {
            if (request.flag == LCI_OK)
            {
                if (request.type == LCI_IOVEC)
                {
                    for (int j = 0; j < request.data.iovec.count; ++j)
                    {
                        LCI_lbuffer_free(request.data.iovec.lbuffers[j]);
                    }
                    free(request.data.iovec.lbuffers);
                    free(request.data.iovec.piggy_back.address);
                }
                else
                {
                    HPX_ASSERT(request.type = LCI_MEDIUM);
                    LCI_mbuffer_free(request.data.mbuffer);
                }
            }
            else
            {
                HPX_ASSERT(request.flag == LCI_ERR_RETRY);
            }
        }
    };

    struct receiver_base
    {
        using buffer_type =
            parcel_buffer<buffer_wrapper, serialization::serialization_chunk>;

        explicit receiver_base(parcelport* pp) noexcept
          : pp_(pp)
        {
        }

        virtual ~receiver_base() {}

        void run() noexcept {}

        virtual bool background_work() noexcept = 0;

    protected:
        parcelport* pp_;
    };

}    // namespace hpx::parcelset::policies::lci

#endif
