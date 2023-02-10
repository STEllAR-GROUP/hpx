//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/serialization/detail/extra_archive_data.hpp>
#include <hpx/serialization/detail/pointer.hpp>
#include <hpx/serialization/input_archive.hpp>
#include <hpx/serialization/output_archive.hpp>

#include <cstdint>
#include <map>
#include <utility>

namespace hpx::serialization {

    namespace detail {

        // This is explicitly instantiated to ensure that the id is stable
        // across shared libraries.
        extra_archive_data_id_type
        extra_archive_data_helper<input_pointer_tracker>::id() noexcept
        {
            static std::uint8_t id = 0;
            return &id;
        }

        extra_archive_data_id_type
        extra_archive_data_helper<output_pointer_tracker>::id() noexcept
        {
            static std::uint8_t id = 0;
            return &id;
        }

        void extra_archive_data_helper<output_pointer_tracker>::reset(
            output_pointer_tracker* data)
        {
            data->clear();
        }
    }    // namespace detail

    void register_pointer(
        input_archive& ar, std::uint64_t pos, detail::ptr_helper_ptr helper)
    {
        auto& tracker = ar.get_extra_data<detail::input_pointer_tracker>();
        HPX_ASSERT(tracker.find(pos) == tracker.end());

        tracker.insert(std::make_pair(pos, HPX_MOVE(helper)));
    }

    detail::ptr_helper& tracked_pointer(input_archive& ar, std::uint64_t pos)
    {
        auto& tracker = ar.get_extra_data<detail::input_pointer_tracker>();

        auto const it = tracker.find(pos);
        HPX_ASSERT(it != tracker.end());

        return *it->second;
    }

    std::uint64_t track_pointer(output_archive& ar, void const* pos)
    {
        auto& tracker = ar.get_extra_data<detail::output_pointer_tracker>();

        auto const it = tracker.find(pos);
        if (it == tracker.end())
        {
            tracker.insert(std::make_pair(pos, ar.bytes_written()));
            return static_cast<std::uint64_t>(-1);
        }
        return it->second;
    }
}    // namespace hpx::serialization
