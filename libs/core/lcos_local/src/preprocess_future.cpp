//  Copyright (c) 2019-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/futures/future.hpp>
#include <hpx/lcos_local/detail/preprocess_future.hpp>
#include <hpx/serialization/detail/extra_archive_data.hpp>
#include <hpx/serialization/output_archive.hpp>

#include <cstdint>

namespace hpx { namespace serialization { namespace detail {

    // This is explicitly instantiated to ensure that the id is stable across
    // shared libraries.
    extra_archive_data_id_type
    extra_archive_data_helper<preprocess_futures>::id() noexcept
    {
        static std::uint8_t id;
        return &id;
    }
}}}    // namespace hpx::serialization::detail

namespace hpx { namespace lcos { namespace detail {

    void preprocess_future(serialization::output_archive& ar,
        hpx::lcos::detail::future_data_refcnt_base& state)
    {
        auto& handle_futures =
            ar.get_extra_data<serialization::detail::preprocess_futures>();

        handle_futures.await_future(state);
    }
}}}    // namespace hpx::lcos::detail
