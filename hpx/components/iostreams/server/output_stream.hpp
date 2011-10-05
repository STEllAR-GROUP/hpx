////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_4AFE0EEA_49F8_4F4C_8945_7B55BF395DA0)
#define HPX_4AFE0EEA_49F8_4F4C_8945_7B55BF395DA0

#include <hpx/config.hpp>
#include <hpx/hpx_fwd.hpp>
#include <hpx/util/spinlock.hpp>
#include <hpx/util/serializable_shared_ptr.hpp>
#include <hpx/components/iostreams/write_functions.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

// TODO: Error handling?

namespace hpx { namespace iostreams { namespace server
{

struct HPX_COMPONENT_EXPORT output_stream
    : components::managed_component_base<output_stream> 
{
    // {{{ types
    typedef components::managed_component_base<output_stream> base_type; 

    typedef hpx::util::spinlock mutex_type;
    // }}}

  private:
    mutex_type mtx;
    write_function_type write_f;

    // Executed in an io_pool thread to prevent io from blocking an HPX
    // shepherd thread.
    void call_write_async(
        util::serializable_shared_ptr<std::deque<char> > const& in
    );

    void call_write_sync(
        util::serializable_shared_ptr<std::deque<char> > const& in
      , threads::thread_id_type caller
    );

  public:
    explicit output_stream(write_function_type write_f_ = write_function_type())
        : write_f(write_f_) {} 

    // STL OutputIterator
    template <typename Iterator>
    output_stream(Iterator it)
        : write_f(make_iterator_write_function(it)) {}

    // std::ostream
    output_stream(boost::reference_wrapper<std::ostream> os)
        : write_f(make_std_ostream_write_function(os.get())) {}

    void write_async(
        util::serializable_shared_ptr<std::deque<char> > const& in
        );

    void write_sync(
        util::serializable_shared_ptr<std::deque<char> > const& in
        );

    enum actions
    {
        output_stream_write_async,
        output_stream_write_sync
    };

    typedef hpx::actions::action1<
        output_stream, output_stream_write_async,
        util::serializable_shared_ptr<std::deque<char> > const&, 
        &output_stream::write_async
    > write_async_action;

    typedef hpx::actions::action1<
        output_stream, output_stream_write_sync,
        util::serializable_shared_ptr<std::deque<char> > const&, 
        &output_stream::write_sync
    > write_sync_action;
};

}}}

#endif // HPX_4AFE0EEA_49F8_4F4C_8945_7B55BF395DA0

