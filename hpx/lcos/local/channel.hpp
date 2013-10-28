////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2012 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_725569E7_7AF4_4276_AF43_5713635DD598)
#define HPX_725569E7_7AF4_4276_AF43_5713635DD598

#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>

namespace hpx { namespace lcos { namespace local
{

/// An asynchronous, single value channel
template <typename T>
struct channel
{
  private:
    typedef hpx::lcos::detail::future_data<T> future_data;

    boost::intrusive_ptr<future_data> data_;

    BOOST_COPYABLE_AND_MOVABLE(channel<T>);

  public:
    typedef typename future_data::completed_callback_type
        completed_callback_type;

    channel() : data_(new future_data()) {}

    channel(channel const& other) : data_(other.data_) {}

    channel(BOOST_RV_REF(channel) other) : data_(boost::move(other.data_)) {}

    explicit channel(BOOST_RV_REF(T) init) : data_(new future_data())
    {
        data_->set_data(init); 
    }

    explicit channel(T const& init) : data_(new future_data())
    {
        data_->set_data(init); 
    }

    ~channel()
    {
        if (data_)
            data_->deleting_owner();
    }

    channel& operator=(BOOST_COPY_ASSIGN_REF(channel) other)
    {
        BOOST_ASSERT(data_);

        if (this != &other)
        {
            data_->deleting_owner();

            data_ = other.data_;
        }

        return *this;
    }

    channel& operator=(BOOST_RV_REF(channel) other)
    {
        BOOST_ASSERT(data_);

        if (this != &other)
        {
            data_->deleting_owner();

            data_ = boost::move(other.data_);
            other.data_.reset();
        }

        return *this;
    }

    void swap(channel& other)
    {
        data_.swap(other.data_);
    }

    void reset()
    {
        BOOST_ASSERT(data_);

        data_->deleting_owner();

        data_->reset();
   }

    T get(hpx::error_code& ec = hpx::throws) const
    {
        BOOST_ASSERT(data_);
        T tmp = data_->get_data(ec);
        return boost::move(tmp);
    }

    T move(hpx::error_code& ec = hpx::throws) const
    {
        BOOST_ASSERT(data_);
        T tmp = data_->move_data(ec);
        return boost::move(tmp);
    }

    hpx::future<T> get_future()
    {
        BOOST_ASSERT(data_);
        return lcos::detail::make_future_from_data<T>(data_);
    }

    void post(BOOST_RV_REF(T) result)
    {
        BOOST_ASSERT(data_);
        //if (data_->is_ready())
        //    data_->move_data();
        data_->set_data(result);
    }

    void post(T const& result)
    {
        BOOST_ASSERT(data_);
        //if (data_->is_ready())
        //    data_->move_data();
        data_->set_data(result);
    }

    template <typename F>
    hpx::future<typename boost::result_of<F(hpx::future<T>)>::type>
    then(BOOST_FWD_REF(F) f)
    {
        BOOST_ASSERT(data_);
        return lcos::detail::make_future_from_data<T>(data_).then
            (boost::forward<F>(f));
    }

    bool is_ready() const
    {
        BOOST_ASSERT(data_);
        return data_->is_ready();
    }
};

template <>
struct channel<void>
{
  private:
    typedef hpx::lcos::detail::future_data<void> future_data;

    boost::intrusive_ptr<future_data> data_;

    BOOST_COPYABLE_AND_MOVABLE(channel<void>);

  public:
    typedef future_data::completed_callback_type
        completed_callback_type;

    channel() : data_(new future_data()) {}

    channel(channel const& other) : data_(other.data_) {}

    channel(BOOST_RV_REF(channel) other) : data_(boost::move(other.data_)) {}

    ~channel()
    {
        if (data_)
            data_->deleting_owner();
    }

    channel& operator=(BOOST_COPY_ASSIGN_REF(channel) other)
    {
        BOOST_ASSERT(data_);

        if (this != &other)
        {
            data_->deleting_owner();

            data_ = other.data_;
        }

        return *this;
    }

    channel& operator=(BOOST_RV_REF(channel) other)
    {
        BOOST_ASSERT(data_);

        if (this != &other)
        {
            data_->deleting_owner();

            data_ = boost::move(other.data_);
            other.data_.reset();
        }

        return *this;
    }

    void swap(channel& other)
    {
        data_.swap(other.data_);
    }

    void reset()
    {
        BOOST_ASSERT(data_);

        data_->deleting_owner();

        data_->reset();
   }

    void get(hpx::error_code& ec = hpx::throws) const
    {
        BOOST_ASSERT(data_);
        hpx::util::unused_type tmp = data_->get_data(ec);
    }

    void move(hpx::error_code& ec = hpx::throws) const
    {
        BOOST_ASSERT(data_);
        hpx::util::unused_type tmp = data_->move_data(ec);
    }

    hpx::future<void> get_future()
    {
        BOOST_ASSERT(data_);
        return lcos::detail::make_future_from_data<void>(data_);
    }

    void post()
    {
        BOOST_ASSERT(data_);
        //if (data_->is_ready())
        //    data_->move_data();
        data_->set_data(hpx::util::unused);
    }

    template <typename F>
    hpx::future<typename boost::result_of<F(hpx::future<void>)>::type>
    then(BOOST_FWD_REF(F) f)
    {
        BOOST_ASSERT(data_);
        return lcos::detail::make_future_from_data<void>(data_).then
            (boost::forward<completed_callback_type>(f));
    }

    bool is_ready() const
    {
        BOOST_ASSERT(data_);
        return data_->is_ready();
    }
};

}}}

#endif // HPX_725569E7_7AF4_4276_AF43_5713635DD598

