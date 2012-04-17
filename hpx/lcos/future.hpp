//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_FUTURE_MAR_06_2012_1059AM)
#define HPX_LCOS_FUTURE_MAR_06_2012_1059AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/detail/future_data.hpp>

#include <boost/move/move.hpp>
#include <boost/intrusive_ptr.hpp>

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    namespace local
    {
        template <typename Result>
        class promise;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Result, typename RemoteResult>
    class future
    {
    private:
        BOOST_COPYABLE_AND_MOVABLE(future)

        typedef lcos::detail::future_data_base<Result, RemoteResult>
            future_data_type;

        future(future_data_type* p)
          : future_data_(p)
        {}

        future(boost::intrusive_ptr<future_data_type> p)
          : future_data_(p)
        {}

        friend class local::promise<Result>;
        friend class promise<Result, RemoteResult>;
        friend class hpx::thread;
        friend struct detail::future_data<Result, RemoteResult>;

    public:

        typedef Result result_type;

        future()
        {}

        ~future()
        {}

        future(future const& other)
          : future_data_(other.future_data_)
        {
        }

        future(BOOST_RV_REF(future) other)
          : future_data_(other.future_data_)
        {
            other.future_data_.reset();
        }

        future& operator=(BOOST_COPY_ASSIGN_REF(future) other)
        {
            future_data_ = other.future_data_;
            return *this;
        }

        future& operator=(BOOST_RV_REF(future) other)
        {
            future_data_ = other.future_data_;
            other.future_data_.reset();
            return *this;
        }

        void swap(future& other)
        {
            future_data_.swap(other.future_data_);
        }

        // retrieving the value
        Result get(error_code& ec = throws) const
        {
            return future_data_->get_data(ec);
        }

        Result move(error_code& ec = throws)
        {
            return future_data_->move_data(ec);
        }

        bool is_ready() const
        {
            return future_data_->is_ready();
        }

        bool has_value() const
        {
            return future_data_->has_value();
        }

        bool has_exception() const
        {
            return future_data_->has_exception();
        }

        future_state::state get_state() const
        {
            if (!future_data_)
                return future_state::uninitialized;

            return future_data_->get_state();
        }

    private:
        boost::intrusive_ptr<future_data_type> future_data_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class future<void, util::unused_type>
    {
    private:
        BOOST_COPYABLE_AND_MOVABLE(future)

        typedef lcos::detail::future_data_base<void, util::unused_type>
            future_data_type;

        future(future_data_type* p)
          : future_data_(p)
        {}

        future(boost::intrusive_ptr<future_data_type> p)
          : future_data_(p)
        {}

        friend class local::promise<void>;
        friend class promise<void, util::unused_type>;
        friend class hpx::thread;
        friend struct detail::future_data<void, util::unused_type>;

    public:
        typedef void result_type;

        future()
        {}

        ~future()
        {}

        future(future const& other)
          : future_data_(other.future_data_)
        {
        }

        future(BOOST_RV_REF(future) other)
          : future_data_(other.future_data_)
        {
            other.future_data_.reset();
        }

        future& operator=(BOOST_COPY_ASSIGN_REF(future) other)
        {
            future_data_ = other.future_data_;
            return *this;
        }

        future& operator=(BOOST_RV_REF(future) other)
        {
            future_data_ = other.future_data_;
            other.future_data_.reset();
            return *this;
        }

        void swap(future& other)
        {
            future_data_.swap(other.future_data_);
        }

        // retrieving the value
        void get(error_code& ec = throws) const
        {
            future_data_->get_data(ec);
        }

        void move(error_code& ec = throws)
        {
            future_data_->move_data(ec);
        }

        bool is_ready() const
        {
            return future_data_->is_ready();
        }

        bool has_value() const
        {
            return future_data_->has_value();
        }

        bool has_exception() const
        {
            return future_data_->has_exception();
        }

        future_state::state get_state() const
        {
            if (!future_data_)
                return future_state::uninitialized;

            return future_data_->get_state();
        }

    private:
        boost::intrusive_ptr<future_data_type> future_data_;
    };
}}

#endif
