//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_SERVER_CHANNEL_JUL_23_731PM)
#define HPX_LCOS_SERVER_CHANNEL_JUL_23_731PM

#include <hpx/config.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/lcos/local/channel.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/traits/get_remote_result.hpp>
#include <hpx/traits/promise_remote_result.hpp>

#include <boost/exception_ptr.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename RemoteType =
        typename traits::promise_remote_result<T>::type>
    class channel;

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename RemoteType>
    class channel
      : public lcos::base_lco_with_value<T, RemoteType>
      , public components::managed_component_base<channel<T, RemoteType> >
    {
    public:
        typedef lcos::base_lco_with_value<T, RemoteType> base_type_holder;

    private:
        typedef components::managed_component_base<channel> base_type;
        typedef typename std::conditional<
            std::is_void<T>::value, util::unused_type, T
        >::type result_type;

    public:
        channel() {}

        // disambiguate base classes
        using base_type::finalize;
        typedef typename base_type::wrapping_type wrapping_type;

        static components::component_type get_component_type()
        {
            return components::get_component_type<channel>();
        }
        static void set_component_type(components::component_type type)
        {
            components::set_component_type<channel>(type);
        }

        // standard LCO action implementations

        // Push a value to the channel.
        void set_value (RemoteType && result)
        {
            channel_.set(std::move(result));
        }

        // Close the channel
        void set_exception(boost::exception_ptr const& /*e*/)
        {
            channel_.close();
        }

        // Retrieve the next value from the channel
        result_type get_value()
        {
            return channel_.get(launch::sync);
        }
        result_type get_value(error_code& ec)
        {
            return channel_.get(launch::sync);
        }

        // Additional functionality exposed by the channel component
        hpx::future<T> get_generation(std::size_t generation)
        {
            return channel_.get(generation);
        }
        HPX_DEFINE_COMPONENT_ACTION(channel, get_generation);

        void set_generation(RemoteType && value, std::size_t generation)
        {
            channel_.set(std::move(value), generation);
        }
        HPX_DEFINE_COMPONENT_ACTION(channel, set_generation);

        void close()
        {
            channel_.close();
        }
        HPX_DEFINE_COMPONENT_ACTION(channel, close);

    private:
        lcos::local::channel<result_type> channel_;
    };
}}}

#define HPX_REGISTER_CHANNEL_DECLARATION(...)                                 \
    HPX_REGISTER_CHANNEL_DECLARATION_(__VA_ARGS__)                            \
/**/
#define HPX_REGISTER_CHANNEL_DECLARATION_(...)                                \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_CHANNEL_DECLARATION_, HPX_UTIL_PP_NARG(__VA_ARGS__)      \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_CHANNEL_DECLARATION_1(type)                              \
    HPX_REGISTER_CHANNEL_DECLARATION_2(type, type)                            \
/**/
#define HPX_REGISTER_CHANNEL_DECLARATION_2(type, name)                        \
    typedef ::hpx::lcos::server::channel<type>                                \
        BOOST_PP_CAT(__channel_, BOOST_PP_CAT(type, name));                   \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        hpx::lcos::server::channel<type>::get_generation_action,              \
        BOOST_PP_CAT(__channel_get_generation_action,                         \
            BOOST_PP_CAT(type, name)));                                       \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        hpx::lcos::server::channel<type>::set_generation_action,              \
        BOOST_PP_CAT(__channel_set_generation_action,                         \
            BOOST_PP_CAT(type, name)));                                       \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        hpx::lcos::server::channel<type>::close_action,                       \
        BOOST_PP_CAT(__channel_close_action,                                  \
            BOOST_PP_CAT(type, name)))                                        \
/**/

#define HPX_REGISTER_CHANNEL(...)                                             \
    HPX_REGISTER_CHANNEL_(__VA_ARGS__)                                        \
/**/
#define HPX_REGISTER_CHANNEL_(...)                                            \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_CHANNEL_, HPX_UTIL_PP_NARG(__VA_ARGS__)                  \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_CHANNEL_1(type)                                          \
    HPX_REGISTER_CHANNEL_2(type, type)                                        \
/**/
#define HPX_REGISTER_CHANNEL_2(type, name)                                    \
    typedef ::hpx::lcos::server::channel<type>                                \
        BOOST_PP_CAT(__channel_, BOOST_PP_CAT(type, name));                   \
    typedef ::hpx::components::managed_component<                             \
            BOOST_PP_CAT(__channel_, BOOST_PP_CAT(type, name))                \
        > BOOST_PP_CAT(__channel_component_, name);                           \
    HPX_REGISTER_DERIVED_COMPONENT_FACTORY(                                   \
        BOOST_PP_CAT(__channel_component_, name),                             \
        BOOST_PP_CAT(__channel_component_, name),                             \
        BOOST_PP_STRINGIZE(BOOST_PP_CAT(__base_lco_with_value_channel_, name)));\
    HPX_REGISTER_ACTION(                                                      \
        hpx::lcos::server::channel<type>::get_generation_action,              \
        BOOST_PP_CAT(__channel_get_generation_action,                         \
            BOOST_PP_CAT(type, name)));                                       \
    HPX_REGISTER_ACTION(                                                      \
        hpx::lcos::server::channel<type>::set_generation_action,              \
        BOOST_PP_CAT(__channel_set_generation_action,                         \
            BOOST_PP_CAT(type, name)));                                       \
    HPX_REGISTER_ACTION(                                                      \
        hpx::lcos::server::channel<type>::close_action,                       \
        BOOST_PP_CAT(__channel_close_action,                                  \
            BOOST_PP_CAT(type, name)))                                        \
/**/

#endif
