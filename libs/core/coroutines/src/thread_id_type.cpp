//  Copyright (c) 2018 Thomas Heller
//  Copyright (c) 2019-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/coroutines/thread_id_type.hpp>
#include <hpx/modules/format.hpp>

#include <cstdio>
#include <ostream>
#include <string_view>

namespace hpx::threads {

    std::ostream& operator<<(std::ostream& os, thread_id const& id)
    {
        os << id.get();
        return os;
    }

    void format_value(
        std::ostream& os, std::string_view spec, thread_id const& id)
    {
        // propagate spec
        char format[16];
        std::snprintf(format, sizeof(format), "{:%.*s}",
            static_cast<int>(spec.size()), spec.data());
        hpx::util::format_to(os, format, id.get());
    }

    namespace detail {

        // reference counting
        void intrusive_ptr_add_ref(thread_data_reference_counting* p) noexcept
        {
            ++p->count_;
        }

        void intrusive_ptr_release(thread_data_reference_counting* p) noexcept
        {
            HPX_ASSERT(p->count_ != 0);
            if (--p->count_ == 0)
            {
                // give this object back to the system
                p->destroy_thread();
            }
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    std::ostream& operator<<(std::ostream& os, thread_id_ref const& id)
    {
        os << id.get();
        return os;
    }

    void format_value(
        std::ostream& os, std::string_view spec, thread_id_ref const& id)
    {
        // propagate spec
        char format[16];
        std::snprintf(
            format, sizeof(format), "{:%.*s}", (int) spec.size(), spec.data());
        hpx::util::format_to(os, format, id.get());
    }
}    // namespace hpx::threads
