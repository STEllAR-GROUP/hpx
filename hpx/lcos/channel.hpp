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
    template <typename T = void>
    class channel
      : public components::client_base<channel<T>, lcos::server::channel<T> >
    {
        typedef components::client_base<
                channel<T>, lcos::server::channel<T>
            > base_type;

    public:
        channel()
        {}

        // create a new instance of a channel component
        channel(naming::id_type const& loc)
          : base_type(hpx::new_<lcos::server::channel<T> >(loc))
        {}
        channel(lcos::future<naming::id_type> f)
          : base_type(std::move(f))
        {}

        hpx::future<T>
        get_async(std::size_t generation = std::size_t(-1)) const
        {
            typedef typename lcos::server::channel<T>::get_generation_action
                action_type;
            return hpx::async(action_type(), this->get_id(), generation);
        }
        T get(std::size_t generation = std::size_t(-1)) const
        {
            return get_async(generation).get();
        }

        template <typename U, typename U2 = T, typename Enable =
            typename std::enable_if<!std::is_void<U2>::value>::type>
        hpx::future<void>
        set_async(U val, std::size_t generation = std::size_t(-1))
        {
            typedef typename lcos::server::channel<T>::set_generation_action
                action_type;
            return hpx::async(action_type(), this->get_id(), std::move(val),
                generation);
        }
        template <typename U, typename U2 = T, typename Enable =
            typename std::enable_if<!std::is_void<U2>::value>::type>
        void set(U val, std::size_t generation = std::size_t(-1))
        {
            typedef typename lcos::server::channel<T>::set_generation_action
                action_type;
            return action_type()(this->get_id(), std::move(val), generation);
        }

        template <typename U = T, typename Enable =
            typename std::enable_if<std::is_void<U>::value>::type>
        hpx::future<void>
        set_async(std::size_t generation = std::size_t(-1))
        {
            typedef typename lcos::server::channel<
                    void
                >::set_generation_action action_type;
            return hpx::async(action_type(), this->get_id(), util::unused,
                generation);
        }
        template <typename U = T, typename Enable =
            typename std::enable_if<std::is_void<U>::value>::type>
        void set(std::size_t generation = std::size_t(-1))
        {
            typedef typename lcos::server::channel<
                    void
                >::set_generation_action action_type;
            return action_type()(this->get_id(), util::unused, generation);
        }

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

    ///////////////////////////////////////////////////////////////////////////
    template <typename T = void>
    class receive_channel
      : public components::client_base<
            receive_channel<T>, lcos::server::channel<T> >
    {
        typedef components::client_base<
                receive_channel<T>, lcos::server::channel<T>
            > base_type;

    public:
        receive_channel()
        {}

        receive_channel(channel<T> const& c)
          : base_type(c.get_id())
        {}

        hpx::future<T>
        get_async(std::size_t generation = std::size_t(-1)) const
        {
            typedef typename lcos::server::channel<T>::get_generation_action
                action_type;
            return hpx::async(action_type(), this->get_id(), generation);
        }
        T get(std::size_t generation = std::size_t(-1)) const
        {
            return get_async(generation).get();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T = void>
    class send_channel
      : public components::client_base<
            send_channel<T>, lcos::server::channel<T> >
    {
        typedef components::client_base<
                send_channel<T>, lcos::server::channel<T>
            > base_type;

    public:
        send_channel()
        {}

        send_channel(channel<T> const& c)
          : base_type(c.get_id())
        {}

        template <typename U, typename U2 = T, typename Enable =
            typename std::enable_if<!std::is_void<U2>::value>::type>
        hpx::future<void>
        set_async(U val, std::size_t generation = std::size_t(-1))
        {
            typedef typename lcos::server::channel<T>::set_generation_action
                action_type;
            return hpx::async(action_type(), this->get_id(), std::move(val),
                generation);
        }
        template <typename U, typename U2 = T, typename Enable =
            typename std::enable_if<!std::is_void<U2>::value>::type>
        void set(U val, std::size_t generation = std::size_t(-1))
        {
            typedef typename lcos::server::channel<T>::set_generation_action
                action_type;
            return action_type()(this->get_id(), std::move(val), generation);
        }

        template <typename U = T, typename Enable =
            typename std::enable_if<std::is_void<U>::value>::type>
        hpx::future<void>
        set_async(std::size_t generation = std::size_t(-1))
        {
            typedef typename lcos::server::channel<
                    void
                >::set_generation_action action_type;
            return hpx::async(action_type(), this->get_id(), util::unused,
                generation);
        }
        template <typename U = T, typename Enable =
            typename std::enable_if<std::is_void<U>::value>::type>
        void set(std::size_t generation = std::size_t(-1))
        {
            typedef typename lcos::server::channel<
                    void
                >::set_generation_action action_type;
            return action_type()(this->get_id(), util::unused, generation);
        }

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
