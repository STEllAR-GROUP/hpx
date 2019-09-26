//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SERIALIZATION_EXTRA_OUTPUT_DATA_HPP)
#define HPX_SERIALIZATION_EXTRA_OUTPUT_DATA_HPP

#include <hpx/datastructures.hpp>
#include <hpx/serialization/extra_archive_data.hpp>

#include <cstddef>

namespace hpx { namespace serialization {

    ////////////////////////////////////////////////////////////////////////////
    constexpr std::size_t extra_output_pointer_tracker = 0;
    constexpr std::size_t default_extra_output_data_size = 1;

    ////////////////////////////////////////////////////////////////////////////
    template <std::size_t N>
    util::moveonly_any_nonser init_extra_output_data_item();

    template <std::size_t N>
    void reset_extra_output_data_item(extra_archive_data_type& data);

    namespace detail {

        ////////////////////////////////////////////////////////////////////////
        // initialize extra data items
        template <std::size_t... Is>
        extra_archive_data_type init_extra_output_data(
            util::detail::pack_c<std::size_t, Is...>)
        {
            extra_archive_data_type data;
            data.reserve(sizeof...(Is));

            int const dummy[] = {0,
                (data.emplace_back(init_extra_output_data_item<Is>()), 0)...};
            (void) dummy;

            return data;
        }

        template <std::size_t Size>
        inline extra_archive_data_type init_extra_output_data()
        {
            return init_extra_output_data(
                util::detail::make_index_pack<Size>{});
        }

        ////////////////////////////////////////////////////////////////////////
        // reset extra data item as needed
        template <std::size_t... Is>
        void reset_extra_output_data(util::detail::pack_c<std::size_t, Is...>,
            extra_archive_data_type& data)
        {
            int const dummy[] = {
                0, (reset_extra_output_data_item<Is>(data), 0)...};
            (void) dummy;
        }

        template <std::size_t Size>
        inline void reset_extra_output_data(extra_archive_data_type& data)
        {
            reset_extra_output_data(
                util::detail::make_index_pack<Size>{}, data);
        }
    }    // namespace detail
}}       // namespace hpx::serialization

#endif
