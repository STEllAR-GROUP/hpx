//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_CHANNEL_JUL_23_2016_0722PM)
#define HPX_LCOS_CHANNEL_JUL_23_2016_0722PM

#include <hpx/config.hpp>
#include <hpx/lcos/server/channel.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/components/client_base.hpp>
#include <hpx/runtime/components/new.hpp>
#include <hpx/runtime/naming_fwd.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T = void> class channel;
    template <typename T = void> class receive_channel;
    template <typename T = void> class send_channel;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Channel>
    class channel_iterator
      : public hpx::util::iterator_facade<
            channel_iterator<T, Channel>, T const, std::input_iterator_tag>
    {
        typedef hpx::util::iterator_facade<
                channel_iterator<T, Channel>, T const, std::input_iterator_tag
            > base_type;

    public:
        channel_iterator()
          : channel_(nullptr), data_(T(), false)
        {}

        explicit channel_iterator(Channel const& c)
          : channel_(&c), data_(get_checked())
        {}

    private:
        std::pair<T, bool> get_checked() const
        {
            hpx::future<T> f = channel_->get();
            f.wait();
            if (!f.has_exception())
            {
                return std::make_pair(f.get(), true);
            }
            return std::make_pair(T(), false);
        }

        friend class hpx::util::iterator_core_access;

        bool equal(channel_iterator const& rhs) const
        {
            return (data_.second == rhs.data_.second &&
                        (channel_ == rhs.channel_ ||
                            (channel_ != nullptr && rhs.channel_ != nullptr &&
                             channel_->get_id() == rhs.channel_->get_id())
                        )
                    ) ||
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
        Channel const* channel_;
        std::pair<T, bool> data_;
    };

    template <typename Channel>
    class channel_iterator<void, Channel>
      : public hpx::util::iterator_facade<
            channel_iterator<void, Channel>, util::unused_type const,
            std::input_iterator_tag>
    {
        typedef hpx::util::iterator_facade<
                channel_iterator<void, Channel>, util::unused_type const,
                std::input_iterator_tag
            > base_type;

    public:
        channel_iterator()
          : channel_(nullptr), data_(false)
        {}

        explicit channel_iterator(Channel const& c)
          : channel_(&c), data_(get_checked())
        {}

    private:
        bool get_checked()
        {
            hpx::future<void> f = channel_->get();
            f.wait();
            return !f.has_exception();
        }

        friend class hpx::util::iterator_core_access;

        bool equal(channel_iterator const& rhs) const
        {
            return (data_ == rhs.data_ &&
                        (channel_ == rhs.channel_ ||
                            (channel_ != nullptr && rhs.channel_ != nullptr &&
                             channel_->get_id() == rhs.channel_->get_id())
                        )
                    ) ||
                (!data_ && rhs.channel_ == nullptr) ||
                (channel_ == nullptr && !rhs.data_);
        }

        void increment()
        {
            if (channel_)
                data_ = get_checked();
        }

        typename base_type::reference dereference() const
        {
            HPX_ASSERT(data_);
            return util::unused;
        }

    private:
        Channel const* channel_;
        bool data_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class channel
      : public components::client_base<channel<T>, lcos::server::channel<T> >
    {
        typedef components::client_base<
                channel<T>, lcos::server::channel<T>
            > base_type;

        HPX_CONSTEXPR_OR_CONST
        static std::size_t default_generation = std::size_t(-1);

    public:
        channel()
        {}

        // create a new instance of a channel component
        explicit channel(naming::id_type const& loc)
          : base_type(hpx::new_<lcos::server::channel<T> >(loc))
        {}
//         channel(lcos::future<naming::id_type> f)
//           : base_type(std::move(f))
//         {}

        ///////////////////////////////////////////////////////////////////////
        hpx::future<T>
        get(std::size_t generation = default_generation) const
        {
            typedef typename lcos::server::channel<T>::get_generation_action
                action_type;
            return hpx::async(action_type(), this->get_id(), generation);
        }
        T get_sync(std::size_t generation = default_generation,
            hpx::error_code& ec = hpx::throws) const
        {
            return get(generation).get(ec);
        }
        T get_sync(hpx::error_code& ec,
            std::size_t generation = default_generation) const
        {
            return get(generation).get(ec);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename U, typename U2 = T, typename Enable =
            typename std::enable_if<!std::is_void<U2>::value>::type>
        hpx::future<void>
        set_async(U val, std::size_t generation = default_generation)
        {
            typedef typename lcos::server::channel<T>::set_generation_action
                action_type;
            return hpx::async(action_type(), this->get_id(), std::move(val),
                generation);
        }
        template <typename U, typename U2 = T, typename Enable =
            typename std::enable_if<!std::is_void<U2>::value>::type>
        void set(U val, std::size_t generation = default_generation)
        {
            typedef typename lcos::server::channel<T>::set_generation_action
                action_type;
            return action_type()(this->get_id(), std::move(val), generation);
        }

        template <typename U = T, typename Enable =
            typename std::enable_if<std::is_void<U>::value>::type>
        hpx::future<void>
        set_async(std::size_t generation = default_generation)
        {
            typedef typename lcos::server::channel<
                    void
                >::set_generation_action action_type;
            return hpx::async(action_type(), this->get_id(), util::unused,
                generation);
        }
        template <typename U = T, typename Enable =
            typename std::enable_if<std::is_void<U>::value>::type>
        void set(std::size_t generation = default_generation)
        {
            typedef typename lcos::server::channel<
                    void
                >::set_generation_action action_type;
            return action_type()(this->get_id(), util::unused, generation);
        }

        ///////////////////////////////////////////////////////////////////////
        hpx::future<void> close_async()
        {
            typedef typename lcos::server::channel<T>::close_action action_type;
            return hpx::async(action_type(), this->get_id());
        }
        void close()
        {
            close_async().get();
        }

        ///////////////////////////////////////////////////////////////////////
        channel_iterator<T, channel<T> > begin() const
        {
            return channel_iterator<T, channel<T> >(*this);
        }
        channel_iterator<T, channel<T> > end() const
        {
            return channel_iterator<T, channel<T> >();
        }

        channel_iterator<T, channel<T> > rbegin() const
        {
            return channel_iterator<T, channel<T> >(*this);
        }
        channel_iterator<T, channel<T> > rend() const
        {
            return channel_iterator<T, channel<T> >();
        }
   };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class receive_channel
      : public components::client_base<
            receive_channel<T>, lcos::server::channel<T> >
    {
        typedef components::client_base<
                receive_channel<T>, lcos::server::channel<T>
            > base_type;

        HPX_CONSTEXPR_OR_CONST
        static std::size_t default_generation = std::size_t(-1);

    public:
        receive_channel()
        {}

        receive_channel(channel<T> const& c)
          : base_type(c.get_id())
        {}

        ///////////////////////////////////////////////////////////////////////
        hpx::future<T> get(std::size_t generation = default_generation) const
        {
            typedef typename lcos::server::channel<T>::get_generation_action
                action_type;
            return hpx::async(action_type(), this->get_id(), generation);
        }
        T get_sync(std::size_t generation = default_generation,
            hpx::error_code& ec = hpx::throws) const
        {
            return get(generation).get(ec);
        }
        T get_sync(hpx::error_code& ec,
            std::size_t generation = default_generation) const
        {
            return get(generation).get(ec);
        }

        ///////////////////////////////////////////////////////////////////////
        channel_iterator<T, channel<T> > begin() const
        {
            return channel_iterator<T, channel<T> >(*this);
        }
        channel_iterator<T, channel<T> > end() const
        {
            return channel_iterator<T, channel<T> >();
        }

        channel_iterator<T, channel<T> > rbegin() const
        {
            return channel_iterator<T, channel<T> >(*this);
        }
        channel_iterator<T, channel<T> > rend() const
        {
            return channel_iterator<T, channel<T> >();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class send_channel
      : public components::client_base<
            send_channel<T>, lcos::server::channel<T> >
    {
        typedef components::client_base<
                send_channel<T>, lcos::server::channel<T>
            > base_type;

        HPX_CONSTEXPR_OR_CONST
        static std::size_t default_generation = std::size_t(-1);

    public:
        send_channel()
        {}

        send_channel(channel<T> const& c)
          : base_type(c.get_id())
        {}

        ///////////////////////////////////////////////////////////////////////
        template <typename U, typename U2 = T, typename Enable =
            typename std::enable_if<!std::is_void<U2>::value>::type>
        hpx::future<void>
        set_async(U val, std::size_t generation = default_generation)
        {
            typedef typename lcos::server::channel<T>::set_generation_action
                action_type;
            return hpx::async(action_type(), this->get_id(), std::move(val),
                generation);
        }
        template <typename U, typename U2 = T, typename Enable =
            typename std::enable_if<!std::is_void<U2>::value>::type>
        void set(U val, std::size_t generation = default_generation)
        {
            typedef typename lcos::server::channel<T>::set_generation_action
                action_type;
            return action_type()(this->get_id(), std::move(val), generation);
        }

        template <typename U = T, typename Enable =
            typename std::enable_if<std::is_void<U>::value>::type>
        hpx::future<void>
        set_async(std::size_t generation = default_generation)
        {
            typedef typename lcos::server::channel<
                    void
                >::set_generation_action action_type;
            return hpx::async(action_type(), this->get_id(), util::unused,
                generation);
        }
        template <typename U = T, typename Enable =
            typename std::enable_if<std::is_void<U>::value>::type>
        void set(std::size_t generation = default_generation)
        {
            typedef typename lcos::server::channel<
                    void
                >::set_generation_action action_type;
            return action_type()(this->get_id(), util::unused, generation);
        }

        ///////////////////////////////////////////////////////////////////////
        hpx::future<void> close_async()
        {
            typedef typename lcos::server::channel<T>::close_action action_type;
            return hpx::async(action_type(), this->get_id());
        }
        void close()
        {
            close_async().get();
        }
    };
}}

#endif
