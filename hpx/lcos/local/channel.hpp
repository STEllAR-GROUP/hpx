//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_LOCAL_CHANNEL_JUL_23_2016_0707PM)
#define HPX_LCOS_LOCAL_CHANNEL_JUL_23_2016_0707PM

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/no_mutex.hpp>
#include <hpx/lcos/local/packaged_task.hpp>
#include <hpx/lcos/local/receive_buffer.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/assert_owns_lock.hpp>
#include <hpx/util/atomic_count.hpp>
#include <hpx/util/iterator_facade.hpp>
#include <hpx/util/register_locks.hpp>
#include <hpx/util/scoped_unlock.hpp>
#include <hpx/util/unused.hpp>

#include <boost/exception_ptr.hpp>
#include <boost/intrusive_ptr.hpp>

#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <mutex>
#include <utility>

namespace hpx { namespace lcos { namespace local
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct channel_impl_base
        {
            channel_impl_base()
              : count_(0)
            {}

            virtual ~channel_impl_base() {}

            virtual hpx::future<T> get(std::size_t generation,
                bool blocking = false) = 0;
            virtual bool try_get(std::size_t generation,
                hpx::future<T>* f = nullptr) = 0;
            virtual hpx::future<void> set(std::size_t generation, T && t) = 0;
            virtual void close() = 0;

            virtual bool requires_delete()
            {
                return 0 == release();
            }
            virtual void destroy()
            {
                delete this;
            }

            long use_count() const { return count_; }
            long addref() { return ++count_; }
            long release() { return --count_; }

        private:
            hpx::util::atomic_count count_;
        };

        // support functions for boost::intrusive_ptr
        template <typename T>
        void intrusive_ptr_add_ref(channel_impl_base<T>* p)
        {
            p->addref();
        }

        template <typename T>
        void intrusive_ptr_release(channel_impl_base<T>* p)
        {
            if (p->requires_delete())
                p->destroy();
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        class unlimited_channel : public channel_impl_base<T>
        {
            typedef hpx::lcos::local::spinlock mutex_type;

            HPX_NON_COPYABLE(unlimited_channel);

        public:
            unlimited_channel()
              : get_generation_(0), set_generation_(0), closed_(false)
            {}

        protected:
            hpx::future<T> get(std::size_t generation, bool blocking)
            {
                std::unique_lock<mutex_type> l(mtx_);

                if (buffer_.empty())
                {
                    if (closed_)
                    {
                        l.unlock();
                        return hpx::make_exceptional_future<T>(
                            HPX_GET_EXCEPTION(hpx::invalid_status,
                                "hpx::lcos::local::channel::get",
                                "this channel is empty and was closed"));
                    }

                    if (blocking && this->use_count() == 1)
                    {
                        l.unlock();
                        return hpx::make_exceptional_future<T>(
                            HPX_GET_EXCEPTION(hpx::invalid_status,
                                "hpx::lcos::local::channel::get",
                                "this channel is empty and is not accessible "
                                "by any other thread causing a deadlock"));
                    }
                }

                ++get_generation_;
                if (generation == std::size_t(-1))
                    generation = get_generation_;

                if (closed_)
                {
                    // the requested item must be available, otherwise this
                    // would create a deadlock
                    hpx::future<T> f;
                    if (!buffer_.try_receive(generation, &f))
                    {
                        l.unlock();
                        return hpx::make_exceptional_future<T>(
                            HPX_GET_EXCEPTION(hpx::invalid_status,
                                "hpx::lcos::local::channel::get",
                                "this channel is closed and the requested value"
                                "has not been received yet"));
                    }
                    return f;
                }

                return buffer_.receive(generation);
            }

            bool try_get(std::size_t generation, hpx::future<T>* f = nullptr)
            {
                std::lock_guard<mutex_type> l(mtx_);

                if (buffer_.empty() && closed_)
                    return false;

                ++get_generation_;
                if (generation == std::size_t(-1))
                    generation = get_generation_;

                if (f != nullptr)
                    *f = buffer_.receive(generation);

                return true;
            }

            hpx::future<void> set(std::size_t generation, T && t)
            {
                std::unique_lock<mutex_type> l(mtx_);
                if(closed_)
                {
                    l.unlock();
                    return hpx::make_exceptional_future<void>(
                        HPX_GET_EXCEPTION(hpx::invalid_status,
                            "hpx::lcos::local::channel::set",
                            "attempting to write to a closed channel"));
                }

                ++set_generation_;
                if (generation == std::size_t(-1))
                    generation = set_generation_;

                buffer_.store_received(generation, std::move(t), &l);
                return hpx::make_ready_future();
            }

            void close()
            {
                std::unique_lock<mutex_type> l(mtx_);
                if(closed_)
                {
                    l.unlock();
                    HPX_THROW_EXCEPTION(hpx::invalid_status,
                        "hpx::lcos::local::channel::close",
                        "attempting to close an already closed channel");
                    return;
                }

                closed_ = true;

                if (buffer_.empty())
                    return;

                boost::exception_ptr e;

                {
                    util::scoped_unlock<std::unique_lock<mutex_type> > ul(l);
                    e = HPX_GET_EXCEPTION(hpx::future_cancelled,
                            "hpx::lcos::local::close",
                            "canceled waiting on this entry");
                }

                // all pending requests which can't be satisfied have to be
                // canceled at this point
                buffer_.cancel_waiting(e);
            }

        private:
            mutable mutex_type mtx_;
            receive_buffer<T, no_mutex> buffer_;
            std::size_t get_generation_;
            std::size_t set_generation_;
            bool closed_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        class one_element_queue_async
        {
            HPX_NON_COPYABLE(one_element_queue_async);

            template <typename T1>
            void set(T1 && val)
            {
                val_ = std::forward<T1>(val);
                empty_ = false;
                push_active_ = false;
            }
            void set_deferred(T && val)
            {
                val_ = std::move(val);
                empty_ = false;
                push_active_ = false;
            }

            T get()
            {
                empty_ = true;
                pop_active_ = false;
                return std::move(val_);
            }

            template <typename T1>
            local::packaged_task<void()> push_pt(T1 && val)
            {
                return local::packaged_task<void()>(
                    util::deferred_call(
                        &one_element_queue_async::set_deferred, this,
                        std::forward<T1>(val)));
            }
            local::packaged_task<T()> pop_pt()
            {
                return local::packaged_task<T()>(
                    util::deferred_call(&one_element_queue_async::get, this));
            }

        public:
            one_element_queue_async()
              : empty_(true), push_active_(false), pop_active_(false)
            {}

            template <typename T1, typename Lock>
            hpx::future<void> push(T1 && val, Lock& l)
            {
                HPX_ASSERT_OWNS_LOCK(l);
                if (!empty_)
                {
                    if (push_active_)
                    {
                        l.unlock();
                        return hpx::make_exceptional_future<void>(
                            HPX_GET_EXCEPTION(hpx::invalid_status,
                                "hpx::lcos::local::detail::"
                                    "one_element_queue_async::push",
                                "attempting to write to a busy queue"));
                    }

                    push_ = push_pt(std::forward<T1>(val));
                    push_active_ = true;
                    return push_.get_future();
                }

                set(std::forward<T1>(val));
                if (pop_active_)
                {
                    pop_();                          // trigger waiting pop
                }
                return hpx::make_ready_future();
            }

            template <typename Lock>
            void cancel(boost::exception_ptr const& e, Lock& l)
            {
                HPX_ASSERT_OWNS_LOCK(l);
                if (pop_active_)
                {
                    pop_.set_exception(e);
                    pop_active_ = false;
                }
            }

            template <typename Lock>
            hpx::future<T> pop(Lock& l)
            {
                HPX_ASSERT_OWNS_LOCK(l);
                if (empty_)
                {
                    if (pop_active_)
                    {
                        l.unlock();
                        return hpx::make_exceptional_future<T>(
                            HPX_GET_EXCEPTION(hpx::invalid_status,
                                "hpx::lcos::local::detail::"
                                    "one_element_queue_async::pop",
                                "attempting to read from an empty queue"));
                    }

                    pop_ = pop_pt();
                    pop_active_ = true;
                    return pop_.get_future();
                }

                T val = get();
                if (push_active_)
                {
                    push_();                        // trigger waiting push
                }
                return hpx::make_ready_future(val);
            }

            template <typename Lock>
            bool is_empty(Lock& l) const
            {
                HPX_ASSERT_OWNS_LOCK(l);
                return empty_;
            }

            template <typename Lock>
            bool has_pending_request(Lock& l) const
            {
                HPX_ASSERT_OWNS_LOCK(l);
                return push_active_;
            }

        private:
            T val_;
            local::packaged_task<void()> push_;
            local::packaged_task<T()> pop_;
            bool empty_;
            bool push_active_;
            bool pop_active_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        class one_element_channel : public channel_impl_base<T>
        {
            typedef hpx::lcos::local::spinlock mutex_type;

            HPX_NON_COPYABLE(one_element_channel);

        public:
            one_element_channel()
              : closed_(false)
            {}

        protected:
            hpx::future<T> get(std::size_t, bool blocking)
            {
                std::unique_lock<mutex_type> l(mtx_);

                if (buffer_.is_empty(l) && !buffer_.has_pending_request(l))
                {
                    if (closed_)
                    {
                        l.unlock();
                        return hpx::make_exceptional_future<T>(
                            HPX_GET_EXCEPTION(hpx::invalid_status,
                                "hpx::lcos::local::channel::get",
                                "this channel is empty and was closed"));
                    }

                    if (blocking && this->use_count() == 1)
                    {
                        l.unlock();
                        return hpx::make_exceptional_future<T>(
                            HPX_GET_EXCEPTION(hpx::invalid_status,
                                "hpx::lcos::local::channel::get",
                                "this channel is empty and is not accessible "
                                "by any other thread causing a deadlock"));
                    }
                }

                hpx::future<T> f = buffer_.pop(l);
                if (closed_ && !f.is_ready())
                {
                    // the requested item must be available, otherwise this
                    // would create a deadlock
                    l.unlock();
                    return hpx::make_exceptional_future<T>(
                        HPX_GET_EXCEPTION(hpx::invalid_status,
                            "hpx::lcos::local::channel::get",
                            "this channel is closed and the requested value"
                            "has not been received yet"));
                }

                return f;
            }

            bool try_get(std::size_t, hpx::future<T>* f = nullptr)
            {
                std::unique_lock<mutex_type> l(mtx_);

                if (buffer_.is_empty(l) && !buffer_.has_pending_request(l) && closed_)
                    return false;

                if (f != nullptr)
                {
                    *f = buffer_.pop(l);
                }
                return true;
            }

            hpx::future<void> set(std::size_t, T && t)
            {
                std::unique_lock<mutex_type> l(mtx_);

                if (closed_)
                {
                    l.unlock();
                    return hpx::make_exceptional_future<void>(
                        HPX_GET_EXCEPTION(hpx::invalid_status,
                            "hpx::lcos::local::channel::set",
                            "attempting to write to a closed channel"));
                }

                return buffer_.push(std::move(t), l);
            }

            void close()
            {
                std::unique_lock<mutex_type> l(mtx_);

                if (closed_)
                {
                    l.unlock();
                    HPX_THROW_EXCEPTION(hpx::invalid_status,
                        "hpx::lcos::local::channel::close",
                        "attempting to close an already closed channel");
                    return;
                }

                closed_ = true;

                if (buffer_.is_empty(l) || !buffer_.has_pending_request(l))
                    return;

                // all pending requests which can't be satisfied have to be
                // canceled at this point
                boost::exception_ptr e;
                {
                    util::scoped_unlock<std::unique_lock<mutex_type> > ul(l);
                    e = boost::exception_ptr(
                        HPX_GET_EXCEPTION(hpx::future_cancelled,
                            "hpx::lcos::local::close",
                            "canceled waiting on this entry"));
                }
                buffer_.cancel(std::move(e), l);
            }

            void set_exception(boost::exception_ptr e)
            {
                std::unique_lock<mutex_type> l(mtx_);
                closed_ = true;

                if (!buffer_.is_empty(l))
                    buffer_.cancel(e, l);
            }

        private:
            mutable mutex_type mtx_;
            one_element_queue_async<T> buffer_;
            bool closed_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T> class channel_base;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T = void> class channel;
    template <typename T = void> class one_element_channel;
    template <typename T = void> class receive_channel;
    template <typename T = void> class send_channel;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class channel_iterator
      : public hpx::util::iterator_facade<
            channel_iterator<T>, T const, std::input_iterator_tag>
    {
        typedef hpx::util::iterator_facade<
                channel_iterator<T>, T const, std::input_iterator_tag
            > base_type;

    public:
        channel_iterator()
          : channel_(nullptr), data_(T(), false)
        {}

        inline explicit channel_iterator(detail::channel_base<T> const* c);
        inline explicit channel_iterator(receive_channel<T> const* c);

    private:
        std::pair<T, bool> get_checked() const
        {
            hpx::future<T> f;
            if (channel_->try_get(std::size_t(-1), &f))
            {
                return std::make_pair(f.get(), true);
            }
            return std::make_pair(T(), false);
        }

        friend class hpx::util::iterator_core_access;

        bool equal(channel_iterator const& rhs) const
        {
            return (channel_ == rhs.channel_ && data_.second == rhs.data_.second) ||
                (!data_.second && rhs.channel_ == nullptr) ||
                (channel_ == nullptr && !rhs.data_.second);
        }

        void increment()
        {
            if (channel_)
                data_ = get_checked();
        }

        typename base_type::reference dereference() const
        {
            HPX_ASSERT(data_.second);
            return data_.first;
        }

    private:
        boost::intrusive_ptr<detail::channel_impl_base<T> > channel_;
        std::pair<T, bool> data_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class channel_async_iterator
      : public hpx::util::iterator_facade<
            channel_async_iterator<T>, hpx::future<T>, std::input_iterator_tag,
            hpx::future<T>>
    {
        typedef hpx::util::iterator_facade<
                channel_async_iterator<T>, hpx::future<T>,
                std::input_iterator_tag, hpx::future<T>
            > base_type;

    public:
        channel_async_iterator()
          : channel_(nullptr), data_(hpx::future<T>(), false)
        {}

        inline explicit channel_async_iterator(detail::channel_base<T> const* c);

    private:
        std::pair<hpx::future<T>, bool> get_checked() const
        {
            hpx::future<T> f;
            if (channel_->try_get(std::size_t(-1), &f))
            {
                return std::make_pair(std::move(f), true);
            }
            return std::make_pair(hpx::future<T>(), false);
        }

        friend class hpx::util::iterator_core_access;

        bool equal(channel_async_iterator const& rhs) const
        {
            return (channel_ == rhs.channel_ && data_.second == rhs.data_.second) ||
                (!data_.second && rhs.channel_ == nullptr) ||
                (channel_ == nullptr && !rhs.data_.second);
        }

        void increment()
        {
            if (channel_)
                data_ = get_checked();
        }

        typename base_type::reference dereference() const
        {
            HPX_ASSERT(data_.second);
            return std::move(data_.first);
        }

    private:
        boost::intrusive_ptr<detail::channel_impl_base<T> > channel_;
        mutable std::pair<hpx::future<T>, bool> data_;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename T>
        class channel_async_range
        {
        public:
            explicit channel_async_range(channel_base<T> const& c)
              : channel_(c)
            {}

            ///////////////////////////////////////////////////////////////////
            channel_async_iterator<T> begin() const
            {
                return channel_async_iterator<T>(&channel_);
            }
            channel_async_iterator<T> end() const
            {
                return channel_async_iterator<T>();
            }

        private:
            channel_base<T> const& channel_;
        };

        template <typename T>
        class channel_base
        {
        protected:
            explicit channel_base(channel_impl_base<T>* impl)
              : channel_(impl)
            {}

        public:
            ///////////////////////////////////////////////////////////////////
            hpx::future<T> get(launch::async_policy,
                std::size_t generation = std::size_t(-1)) const
            {
                return channel_->get(generation);
            }
            hpx::future<T> get(std::size_t generation = std::size_t(-1)) const
            {
                return get(launch::async, generation);
            }

            T get(launch::sync_policy, std::size_t generation = std::size_t(-1),
                error_code& ec = throws) const
            {
                return channel_->get(generation, true).get(ec);
            }
            T get(launch::sync_policy, error_code& ec,
                std::size_t generation = std::size_t(-1)) const
            {
                return channel_->get(generation, true).get(ec);
            }

            ///////////////////////////////////////////////////////////////////
            void set(T val, std::size_t generation = std::size_t(-1))
            {
                channel_->set(generation, std::move(val)).get();
            }
            void set(launch::sync_policy, T val,
                std::size_t generation = std::size_t(-1))
            {
                channel_->set(generation, std::move(val)).get();
            }
            hpx::future<void> set(launch::async_policy, T val,
                std::size_t generation = std::size_t(-1))
            {
                return channel_->set(generation, std::move(val));
            }

            void close()
            {
                channel_->close();
            }

            ///////////////////////////////////////////////////////////////////
            channel_iterator<T> begin() const
            {
                return channel_iterator<T>(this);
            }
            channel_iterator<T> end() const
            {
                return channel_iterator<T>();
            }

            channel_base const& range() const
            {
                return *this;
            }
            channel_base const& range(launch::sync_policy) const
            {
                return *this;
            }
            channel_async_range<T> range(launch::async_policy) const
            {
                return channel_async_range<T>(*this);
            }

            ///////////////////////////////////////////////////////////////////
            channel_impl_base<T>* get_channel_impl() const
            {
                return channel_.get();
            }

        protected:
            boost::intrusive_ptr<channel_impl_base<T> > channel_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    // channel with unlimited buffer
    template <typename T>
    class channel : protected detail::channel_base<T>
    {
        typedef detail::channel_base<T> base_type;

    private:
        friend class channel_iterator<T>;
        friend class receive_channel<T>;
        friend class send_channel<T>;

    public:
        typedef T value_type;

        channel()
          : base_type(new detail::unlimited_channel<T>())
        {}

        using base_type::get;
        using base_type::set;
        using base_type::close;
        using base_type::begin;
        using base_type::end;
        using base_type::range;
    };

    // channel with a one-element buffer
    template <typename T>
    class one_element_channel : protected detail::channel_base<T>
    {
        typedef detail::channel_base<T> base_type;

    private:
        friend class channel_iterator<T>;
        friend class receive_channel<T>;
        friend class send_channel<T>;

    public:
        typedef T value_type;

        one_element_channel()
          : base_type(new detail::one_element_channel<T>())
        {}

        using base_type::get;
        using base_type::set;
        using base_type::close;
        using base_type::begin;
        using base_type::end;
        using base_type::range;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class receive_channel : protected detail::channel_base<T>
    {
        typedef detail::channel_base<T> base_type;

    private:
        friend class channel_iterator<T>;
        friend class send_channel<T>;

    public:
        receive_channel(channel<T> const& c)
          : base_type(c.get_channel_impl())
        {}
        receive_channel(one_element_channel<T> const& c)
          : base_type(c.get_channel_impl())
        {}

        using base_type::get;
        using base_type::begin;
        using base_type::end;
        using base_type::range;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class send_channel : private detail::channel_base<T>
    {
        typedef detail::channel_base<T> base_type;

    public:
        send_channel(channel<T> const& c)
          : base_type(c.get_channel_impl())
        {}
        send_channel(one_element_channel<T> const& c)
          : base_type(c.get_channel_impl())
        {}

        using base_type::set;
        using base_type::close;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    inline channel_iterator<T>::channel_iterator(detail::channel_base<T> const* c)
      : channel_(c ? c->get_channel_impl() : nullptr),
        data_(c ? get_checked() : std::make_pair(T(), false))
    {}

    template <typename T>
    inline channel_iterator<T>::channel_iterator(receive_channel<T> const* c)
      : channel_(c ? c->get_channel_impl() : nullptr),
        data_(c ? get_checked() : std::make_pair(T(), false))
    {}

    template <typename T>
    inline channel_async_iterator<T>::channel_async_iterator(
            detail::channel_base<T> const* c)
      : channel_(c ? c->get_channel_impl() : nullptr),
        data_(c ? get_checked() : std::make_pair(hpx::future<T>(), false))
    {}

    ///////////////////////////////////////////////////////////////////////////
    // forward declare specializations
    template <> class channel<void>;
    template <> class receive_channel<void>;
    template <> class send_channel<void>;

    template <>
    class channel_iterator<void>
      : public hpx::util::iterator_facade<
            channel_iterator<void>, util::unused_type const,
            std::input_iterator_tag>
    {
        typedef hpx::util::iterator_facade<
                channel_iterator<void>, util::unused_type const,
                std::input_iterator_tag
            > base_type;

    public:
        channel_iterator()
          : channel_(nullptr), data_(false)
        {}

        inline explicit channel_iterator(detail::channel_base<void> const* c);
        inline explicit channel_iterator(receive_channel<void> const* c);

    private:
        bool get_checked()
        {
            hpx::future<util::unused_type> f;
            if (channel_->try_get(std::size_t(-1), &f))
            {
                f.get();
                return true;
            }
            return false;
        }

        friend class hpx::util::iterator_core_access;

        bool equal(channel_iterator const& rhs) const
        {
            return (channel_ == rhs.channel_ && data_ == rhs.data_) ||
                (!data_ && rhs.channel_ == nullptr) ||
                (channel_ == nullptr && !rhs.data_);
        }

        void increment()
        {
            if (channel_)
                data_ = get_checked();
        }

        base_type::reference dereference() const
        {
            HPX_ASSERT(data_);
            return util::unused;
        }

    private:
        boost::intrusive_ptr<detail::channel_impl_base<util::unused_type> > channel_;
        bool data_;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <>
        class channel_base<void>
        {
        public:
            explicit channel_base(detail::channel_impl_base<util::unused_type>* impl)
              : channel_(impl)
            {}

            ///////////////////////////////////////////////////////////////////////
            hpx::future<void> get(launch::async_policy,
                std::size_t generation = std::size_t(-1)) const
            {
                return channel_->get(generation);
            }
            hpx::future<void> get(std::size_t generation = std::size_t(-1)) const
            {
                return get(launch::async, generation);
            }
            void get(launch::sync_policy, std::size_t generation = std::size_t(-1),
                error_code& ec = throws) const
            {
                channel_->get(generation, true).get(ec);
            }
            void get(launch::sync_policy, error_code& ec,
                std::size_t generation = std::size_t(-1)) const
            {
                channel_->get(generation, true).get(ec);
            }

            ///////////////////////////////////////////////////////////////////////
            void set(std::size_t generation = std::size_t(-1))
            {
                channel_->set(generation, hpx::util::unused_type()).get();
            }
            void set(launch::sync_policy,
                std::size_t generation = std::size_t(-1))
            {
                channel_->set(generation, hpx::util::unused_type()).get();
            }
            hpx::future<void> set(launch::async_policy,
                std::size_t generation = std::size_t(-1))
            {
                return channel_->set(generation, hpx::util::unused_type());
            }

            void close()
            {
                channel_->close();
            }

            ///////////////////////////////////////////////////////////////////
            channel_iterator<void> begin() const
            {
                return channel_iterator<void>(this);
            }
            channel_iterator<void> end() const
            {
                return channel_iterator<void>();
            }

            channel_base const& range() const
            {
                return *this;
            }
            channel_base const& range(launch::sync_policy) const
            {
                return *this;
            }
            channel_async_range<void> range(launch::async_policy) const
            {
                return channel_async_range<void>(*this);
            }

            ///////////////////////////////////////////////////////////////////
            channel_impl_base<util::unused_type>* get_channel_impl() const
            {
                return channel_.get();
            }

        protected:
            boost::intrusive_ptr<channel_impl_base<util::unused_type> > channel_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class channel<void> : protected detail::channel_base<void>
    {
        typedef detail::channel_base<void> base_type;

    private:
        friend class channel_iterator<void>;
        friend class receive_channel<void>;
        friend class send_channel<void>;

    public:
        typedef void value_type;

        channel()
          : base_type(new detail::unlimited_channel<util::unused_type>())
        {}

        using base_type::get;
        using base_type::set;
        using base_type::close;
        using base_type::begin;
        using base_type::end;
        using base_type::range;
    };

    template <>
    class one_element_channel<void> : protected detail::channel_base<void>
    {
        typedef detail::channel_base<void> base_type;

    private:
        friend class channel_iterator<void>;
        friend class receive_channel<void>;
        friend class send_channel<void>;

    public:
        typedef void value_type;

        one_element_channel()
          : base_type(new detail::one_element_channel<util::unused_type>())
        {}

        using base_type::get;
        using base_type::set;
        using base_type::close;
        using base_type::begin;
        using base_type::end;
        using base_type::range;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class receive_channel<void> : protected detail::channel_base<void>
    {
        typedef detail::channel_base<void> base_type;

    private:
        friend class channel_iterator<void>;
        friend class send_channel<void>;

    public:
        receive_channel(channel<void> const& c)
          : base_type(c.get_channel_impl())
        {}
        receive_channel(one_element_channel<void> const& c)
          : base_type(c.get_channel_impl())
        {}

        using base_type::get;
        using base_type::begin;
        using base_type::end;
        using base_type::range;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    class send_channel<void> : private detail::channel_base<void>
    {
        typedef detail::channel_base<void> base_type;

    public:
        send_channel(channel<void> const& c)
          : base_type(c.get_channel_impl())
        {}
        send_channel(one_element_channel<void> const& c)
          : base_type(c.get_channel_impl())
        {}

        using base_type::set;
        using base_type::close;
    };

    ///////////////////////////////////////////////////////////////////////////
    inline channel_iterator<void>::channel_iterator(detail::channel_base<void> const* c)
      : channel_(c ? c->get_channel_impl() : nullptr),
        data_(c ? get_checked() : false)
    {}

    inline channel_iterator<void>::channel_iterator(receive_channel<void> const* c)
      : channel_(c ? c->get_channel_impl() : nullptr),
        data_(c ? get_checked() : false)
    {}
}}}

#endif
