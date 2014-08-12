//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_4AFE0EEA_49F8_4F4C_8945_7B55BF395DA0)
#define HPX_4AFE0EEA_49F8_4F4C_8945_7B55BF395DA0

#include <hpx/hpx_fwd.hpp>
#include <hpx/components/iostreams/write_functions.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include <hpx/components/iostreams/server/buffer.hpp>
#include <hpx/components/iostreams/server/order_output.hpp>

// TODO: Error handling?

namespace hpx { namespace iostreams { namespace server
{
    struct HPX_COMPONENT_EXPORT output_stream
      : components::managed_component_base<output_stream>
    {
        // {{{ types
        typedef components::managed_component_base<output_stream> base_type;
        typedef lcos::local::spinlock mutex_type;
        // }}}

    private:
        mutable mutex_type mtx_;
        write_function_type write_f;
        detail::order_output pending_output_;

        // Executed in an io_pool thread to prevent io from blocking an HPX
        // shepherd thread.
        void call_write_async(boost::uint32_t locality_id, boost::uint64_t count,
            detail::buffer in);
        void call_write_sync(boost::uint32_t locality_id, boost::uint64_t count,
            detail::buffer in, threads::thread_id_type caller);

    public:
        explicit output_stream(write_function_type write_f_ = write_function_type())
          : write_f(write_f_)
        {}

        // STL OutputIterator
        template <typename Iterator>
        output_stream(Iterator it)
          : write_f(make_iterator_write_function(it))
        {}

        // std::ostream
        output_stream(boost::reference_wrapper<std::ostream> os)
          : write_f(make_std_ostream_write_function(os.get()))
        {}

        void write_async(boost::uint32_t locality_id,
            boost::uint64_t count, detail::buffer const& in);
        void write_sync(boost::uint32_t locality_id,
            boost::uint64_t count, detail::buffer const& in);

        HPX_DEFINE_COMPONENT_ACTION(output_stream, write_async);
        HPX_DEFINE_COMPONENT_ACTION(output_stream, write_sync);
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

#endif // HPX_4AFE0EEA_49F8_4F4C_8945_7B55BF395DA0

