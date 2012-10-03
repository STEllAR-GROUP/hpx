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
#include <hpx/components/iostreams/write_functions.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/component_action.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/deque.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/split_member.hpp>

// TODO: Error handling?

namespace hpx { namespace iostreams
{

struct buffer
{
    buffer()
      : data_()
    {}

    buffer(
        std::deque<char>* ptr
        )
      : data_(ptr)
    {}

    boost::shared_ptr<std::deque<char> > data_;

  private:
    friend class boost::serialization::access;

    template <
        typename Archive
    >
    void save(
        Archive& ar
      , const unsigned int
        ) const
    {
        bool isvalid = data_ ? true : false;
        ar << isvalid;

        if (isvalid)
        {
            std::deque<char> const& instance = *data_;
            ar << instance;
        }
    }

    template <
        typename Archive
    >
    void load(
        Archive& ar
      , const unsigned int
        )
    {
        bool isvalid;
        ar >> isvalid;

        if (isvalid)
        {
            std::deque<char> instance;
            ar >> instance;
            data_.reset(new std::deque<char>(instance));
        }
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

namespace server
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
        buffer const& in
        );

    void call_write_sync(
        buffer const& in
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
        buffer const& in
        );

    void write_sync(
        buffer const& in
        );

    enum actions
    {
        output_stream_write_async,
        output_stream_write_sync
    };

    typedef hpx::actions::action1<
        output_stream, output_stream_write_async,
        buffer const&,
        &output_stream::write_async
    > write_async_action;

    typedef hpx::actions::action1<
        output_stream, output_stream_write_sync,
        buffer const&,
        &output_stream::write_sync
    > write_sync_action;
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

