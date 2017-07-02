//  Copyright (c) 2016-2017 Hartmut Kaiser
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
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming_fwd.hpp>

#include <cstddef>
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
    namespace detail
    {
        template <typename T, bool Migratable>
        class channel_base
          : public components::client_base<
                channel_base<T, Migratable>,
                lcos::server::channel<T, Migratable> >
        {
        protected:
            typedef components::client_base<
                    channel_base, lcos::server::channel<T, Migratable>
                > base_type;

            HPX_CONSTEXPR_OR_CONST
            static std::size_t default_generation = std::size_t(-1);

        public:
            // FIXME: this typedef is currently needed for traits::is_future_range
            // to work properly. This typedef is to workaround that defect in the
            // trait implementation which expects the value_type typedef once begin
            // and end members are present.
            typedef T value_type;

            channel_base() = default;

            // create a new instance of a channel component
            explicit channel_base(naming::id_type const& loc)
              : base_type(hpx::new_<lcos::server::channel<T, Migratable> >(loc))
            {}

            explicit channel_base(hpx::future<naming::id_type>&& id)
              : base_type(std::move(id))
            {}

            explicit channel_base(hpx::shared_future<naming::id_type>&& id)
              : base_type(std::move(id))
            {}

            explicit channel_base(hpx::shared_future<naming::id_type> const& id)
              : base_type(id)
            {}

            ///////////////////////////////////////////////////////////////////
            hpx::future<T>
            get(launch::async_policy,
                std::size_t generation = default_generation) const
            {
                typedef typename lcos::server::channel<
                        T, Migratable
                    >::get_generation_action action_type;
                return hpx::async(action_type(), this->get_id(), generation);
            }
            hpx::future<T>
            get(std::size_t generation = default_generation) const
            {
                return get(launch::async, generation);
            }
            T get(launch::sync_policy, std::size_t generation = default_generation,
                hpx::error_code& ec = hpx::throws) const
            {
                return get(generation).get(ec);
            }
            T get(launch::sync_policy, hpx::error_code& ec,
                std::size_t generation = default_generation) const
            {
                return get(launch::sync, generation, ec);
            }

            ///////////////////////////////////////////////////////////////////
            template <typename U, typename U2 = T>
            typename std::enable_if<!std::is_void<U2>::value, bool>::type
            set(launch::apply_policy, U val,
                std::size_t generation = default_generation)
            {
                typedef typename lcos::server::channel<
                        T, Migratable
                    >::set_generation_action action_type;
                return hpx::apply(action_type(), this->get_id(), std::move(val),
                    generation);
            }
            template <typename U, typename U2 = T>
            typename std::enable_if<
                !std::is_void<U2>::value, hpx::future<void>
            >::type
            set(launch::async_policy, U val,
                std::size_t generation = default_generation)
            {
                typedef typename lcos::server::channel<T>::set_generation_action
                    action_type;
                return hpx::async(action_type(), this->get_id(), std::move(val),
                    generation);
            }
            template <typename U, typename U2 = T>
            typename std::enable_if<!std::is_void<U2>::value>::type
            set(launch::sync_policy, U val,
                std::size_t generation = default_generation)
            {
                typedef typename lcos::server::channel<
                        T, Migratable
                    >::set_generation_action action_type;
                action_type()(this->get_id(), std::move(val), generation);
            }
            template <typename U, typename U2 = T>
            typename std::enable_if<
                !std::is_void<U2>::value &&
                !traits::is_launch_policy<U>::value
            >::type
            set(U val, std::size_t generation = default_generation)
            {
                set(launch::sync, std::move(val), generation);
            }

            template <typename U = T>
            typename std::enable_if<std::is_void<U>::value, bool>::type
            set(launch::apply_policy, std::size_t generation = default_generation)
            {
                typedef typename lcos::server::channel<
                        void, Migratable
                    >::set_generation_action action_type;
                return hpx::apply(action_type(), this->get_id(), util::unused,
                    generation);
            }
            template <typename U = T>
            typename std::enable_if<
                std::is_void<U>::value, hpx::future<void>
            >::type
            set(launch::async_policy, std::size_t generation = default_generation)
            {
                typedef typename lcos::server::channel<
                        void, Migratable
                    >::set_generation_action action_type;
                return hpx::async(action_type(), this->get_id(), util::unused,
                    generation);
            }
            template <typename U = T>
            typename std::enable_if<std::is_void<U>::value>::type
            set(launch::sync_policy, std::size_t generation = default_generation)
            {
                typedef typename lcos::server::channel<
                        void, Migratable
                    >::set_generation_action action_type;
                return action_type()(this->get_id(), util::unused, generation);
            }
            template <typename U = T>
            typename std::enable_if<std::is_void<U>::value>::type
            set(std::size_t generation = default_generation)
            {
                set(launch::sync, generation);
            }

            ///////////////////////////////////////////////////////////////////
            void close(launch::apply_policy)
            {
                typedef typename lcos::server::channel<
                        T, Migratable
                    >::close_action action_type;
                hpx::apply(action_type(), this->get_id());
            }
            hpx::future<void> close(launch::async_policy)
            {
                typedef typename lcos::server::channel<
                        T, Migratable
                    >::close_action action_type;
                return hpx::async(action_type(), this->get_id());
            }
            void close(launch::sync_policy)
            {
                typedef typename lcos::server::channel<
                        T, Migratable
                    >::close_action action_type;
                action_type()(this->get_id());
            }
            void close()
            {
                close(launch::sync);
            }

            ///////////////////////////////////////////////////////////////////
            channel_iterator<T, channel_base> begin() const
            {
                return channel_iterator<T, channel_base>(*this);
            }
            channel_iterator<T, channel_base> end() const
            {
                return channel_iterator<T, channel_base>();
            }

            channel_iterator<T, channel_base> rbegin() const
            {
                return channel_iterator<T, channel_base>(*this);
            }
            channel_iterator<T, channel_base> rend() const
            {
                return channel_iterator<T, channel_base>();
            }
       };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class channel : public detail::channel_base<T, false>
    {
    private:
        typedef detail::channel_base<T, false> base_type;

    public:
        channel() = default;

        // create a new instance of a channel component
        explicit channel(naming::id_type const& loc)
          : base_type(loc)
        {}

        explicit channel(hpx::future<naming::id_type>&& id)
          : base_type(std::move(id))
        {}

        explicit channel(hpx::shared_future<naming::id_type>&& id)
          : base_type(std::move(id))
        {}

        explicit channel(hpx::shared_future<naming::id_type> const& id)
          : base_type(id)
        {}

        typedef typename base_type::value_type value_type;

        using base_type::get_id;

        using base_type::get;
        using base_type::set;
        using base_type::close;

        using base_type::begin;
        using base_type::end;
        using base_type::rbegin;
        using base_type::rend;
    };

    template <typename T>
    class receive_channel : public detail::channel_base<T, false>
    {
    private:
        typedef detail::channel_base<T, false> base_type;

        using base_type::set;
        using base_type::close;

    public:
        receive_channel() = default;

        receive_channel(channel<T> const& c)
          : base_type(c)
        {}

        explicit receive_channel(hpx::future<naming::id_type>&& id)
          : base_type(std::move(id))
        {}

        explicit receive_channel(hpx::shared_future<naming::id_type>&& id)
          : base_type(std::move(id))
        {}

        explicit receive_channel(hpx::shared_future<naming::id_type> const& id)
          : base_type(id)
        {}

        typedef typename base_type::value_type value_type;

        using base_type::get_id;

        using base_type::get;

        using base_type::begin;
        using base_type::end;
        using base_type::rbegin;
        using base_type::rend;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class send_channel : public detail::channel_base<T, false>
    {
    private:
        typedef detail::channel_base<T, false> base_type;

        using base_type::get;

        using base_type::begin;
        using base_type::end;
        using base_type::rbegin;
        using base_type::rend;

    public:
        send_channel() = default;

        send_channel(channel<T> const& c)
          : base_type(c)
        {}

        explicit send_channel(hpx::future<naming::id_type>&& id)
          : base_type(std::move(id))
        {}

        explicit send_channel(hpx::shared_future<naming::id_type>&& id)
          : base_type(std::move(id))
        {}

        explicit send_channel(hpx::shared_future<naming::id_type> const& id)
          : base_type(id)
        {}

        typedef typename base_type::value_type value_type;

        using base_type::get_id;

        using base_type::set;
        using base_type::close;
    };
}}

#endif
