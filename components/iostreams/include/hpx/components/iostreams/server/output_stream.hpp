//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/actions/transfer_action.hpp>
#include <hpx/actions_base/component_action.hpp>
#include <hpx/async_distributed/transfer_continuation_action.hpp>
#include <hpx/components_base/server/component_base.hpp>
#include <hpx/serialization/vector.hpp>

#include <hpx/components/iostreams/export_definitions.hpp>
#include <hpx/components/iostreams/server/buffer.hpp>
#include <hpx/components/iostreams/server/order_output.hpp>
#include <hpx/components/iostreams/write_functions.hpp>

#include <cstdint>
#include <functional>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace iostreams { namespace server
{
    struct HPX_IOSTREAMS_EXPORT output_stream
      : components::component_base<output_stream>
    {
        // {{{ types
        typedef components::component_base<output_stream> base_type;
        typedef lcos::local::spinlock mutex_type;
        // }}}

    private:
        mutable mutex_type mtx_;
        write_function_type write_f;
        detail::order_output pending_output_;

        // Executed in an io_pool thread to prevent io from blocking an HPX
        // shepherd thread.
        void call_write_async(std::uint32_t locality_id, std::uint64_t count,
            detail::buffer const& in, hpx::id_type /*this_id*/);
        void call_write_sync(std::uint32_t locality_id, std::uint64_t count,
            detail::buffer const& in, threads::thread_id_ref_type caller);

    public:
        explicit output_stream(write_function_type write_f_ = write_function_type())
          : write_f(write_f_)
        {}

        // STL OutputIterator
        template <typename Iterator>
        explicit output_stream(Iterator it)
          : write_f(make_iterator_write_function(it))
        {}

        // std::ostream
        explicit output_stream(std::ostream& os)
          : write_f(make_std_ostream_write_function(os))
        {}

        explicit output_stream(std::reference_wrapper<std::ostream> const& os)
          : write_f(make_std_ostream_write_function(os.get()))
        {}

        void write_async(std::uint32_t locality_id,
            std::uint64_t count, detail::buffer const& in);
        void write_sync(std::uint32_t locality_id,
            std::uint64_t count, detail::buffer const& in);

        HPX_DEFINE_COMPONENT_ACTION(output_stream, write_async)
        HPX_DEFINE_COMPONENT_ACTION(output_stream, write_sync)
    };
}}}

HPX_REGISTER_ACTION_DECLARATION(
    hpx::iostreams::server::output_stream::write_async_action
  , output_stream_write_async_action
)

HPX_REGISTER_ACTION_DECLARATION(
    hpx::iostreams::server::output_stream::write_sync_action
  , output_stream_write_sync_action
)

#include <hpx/config/warnings_suffix.hpp>


