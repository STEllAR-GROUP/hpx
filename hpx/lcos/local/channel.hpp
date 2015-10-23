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

#include <boost/thread/locks.hpp>

namespace hpx { namespace lcos { namespace detail
{

template <typename Result>
struct channel_future_data : future_data<Result>
{
  public:
    typedef typename future_data<Result>::mutex_type mutex_type;
    typedef typename future_data<Result>::result_type result_type;

  public:
    result_type move_data(error_code& ec = throws)
    {
        typedef typename future_data<Result>::result_type data_type;
        return std::move(*this->get_result(ec));
    }
};

} // namespace detail

namespace local
{

/// An asynchronous, single value channel
template <typename T>
struct channel
{
  private:
    typedef hpx::lcos::detail::channel_future_data<T> future_data;

    boost::intrusive_ptr<future_data> data_;

  public:
    typedef typename future_data::completed_callback_type
        completed_callback_type;

    channel() : data_(new future_data()) {}

    channel(channel const& other) : data_(other.data_) {}

    channel(channel && other) : data_(std::move(other.data_)) {}

    explicit channel(T && init) : data_(new future_data())
    {
        data_->set_data(init);
    }

    explicit channel(T const& init) : data_(new future_data())
    {
        data_->set_data(init);
    }

    ~channel()
    {}

    channel& operator=(channel const & other)
    {
        HPX_ASSERT(data_);

        if (this != &other)
        {
            data_ = other.data_;
        }

        return *this;
    }

    channel& operator=(channel && other)
    {
        HPX_ASSERT(data_);

        if (this != &other)
        {
            data_ = std::move(other.data_);
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
        HPX_ASSERT(data_);

        data_->reset();
   }

    hpx::future<T> get_future()
    {
        HPX_ASSERT(data_);

        using traits::future_access;
        return future_access<hpx::future<T> >::create(data_);
    }

    T get(hpx::error_code& ec = hpx::throws) const
    {
        HPX_ASSERT(data_);
        T tmp = data_->get_data(ec);
        return tmp;
    }

    T move(hpx::error_code& ec = hpx::throws) const
    {
        HPX_ASSERT(data_);
        T tmp = data_->move_data(ec);
        return tmp;
    }

    void post(T && result)
    {
        HPX_ASSERT(data_);
        //if (data_->is_ready())
        //    data_->move_data();
        data_->set_data(result);
    }

    void post(T const& result)
    {
        HPX_ASSERT(data_);
        //if (data_->is_ready())
        //    data_->move_data();
        data_->set_data(result);
    }

    template <typename F>
    hpx::future<typename util::result_of<F(hpx::future<T>)>::type>
    then(F && f)
    {
        HPX_ASSERT(data_);

        using traits::future_access;
        return future_access<hpx::future<T> >::create(data_).then
            (std::forward<F>(f));
    }

    bool is_ready() const
    {
        HPX_ASSERT(data_);
        return data_->is_ready();
    }
};

template <>
struct channel<void>
{
  private:
    typedef hpx::lcos::detail::channel_future_data<void> future_data;

    boost::intrusive_ptr<future_data> data_;

  public:
    typedef future_data::completed_callback_type
        completed_callback_type;

    channel() : data_(new future_data()) {}

    channel(channel const& other) : data_(other.data_) {}

    channel(channel && other) : data_(std::move(other.data_)) {}

    ~channel()
    {}

    channel& operator=(channel const & other)
    {
        HPX_ASSERT(data_);

        if (this != &other)
        {
            data_ = other.data_;
        }

        return *this;
    }

    channel& operator=(channel && other)
    {
        HPX_ASSERT(data_);

        if (this != &other)
        {
            data_ = std::move(other.data_);
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
        HPX_ASSERT(data_);

        data_->reset();
   }

    hpx::future<void> get_future()
    {
        HPX_ASSERT(data_);

        using traits::future_access;
        return future_access<hpx::future<void> >::create(data_);
    }

    void get(hpx::error_code& ec = hpx::throws) const
    {
        HPX_ASSERT(data_);
        hpx::util::unused_type tmp = data_->get_result(ec);
    }

    void move(hpx::error_code& ec = hpx::throws) const
    {
        HPX_ASSERT(data_);
        hpx::util::unused_type tmp = data_->move_data(ec);
    }

    void post()
    {
        HPX_ASSERT(data_);
        //if (data_->is_ready())
        //    data_->move_data();
        data_->set_data(hpx::util::unused);
    }

    template <typename F>
    hpx::future<typename util::result_of<F(hpx::future<void>)>::type>
    then(F && f)
    {
        HPX_ASSERT(data_);

        using traits::future_access;
        return future_access<hpx::future<void> >::create(data_).then
            (std::forward<completed_callback_type>(f));
    }

    bool is_ready() const
    {
        HPX_ASSERT(data_);
        return data_->is_ready();
    }
};

}}}

#endif // HPX_725569E7_7AF4_4276_AF43_5713635DD598

