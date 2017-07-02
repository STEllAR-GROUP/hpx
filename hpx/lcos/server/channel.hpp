//  Copyright (c) 2016-2017 Hartmut Kaiser
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
#include <hpx/runtime/components/server/migration_support.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/traits/get_remote_result.hpp>
#include <hpx/traits/is_component.hpp>
#include <hpx/traits/promise_remote_result.hpp>

#include <cstddef>
#include <exception>
#include <type_traits>
#include <utility>

#include <boost/preprocessor/cat.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, bool Migratable = false, typename RemoteType =
        typename traits::promise_remote_result<T>::type>
    class channel;

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename RemoteType>
        class channel_base
          : public lcos::base_lco_with_value<
                T, RemoteType, traits::detail::simple_component_tag>
        {
        public:
            typedef lcos::base_lco_with_value<
                    T, RemoteType, traits::detail::simple_component_tag
                > base_type_holder;

        private:
            typedef typename std::conditional<
                    std::is_void<T>::value, util::unused_type, T
                >::type result_type;

        public:
            channel_base() = default;

            // standard LCO action implementations

            // Push a value to the channel.
            void set_value (RemoteType && result)
            {
                channel_.set(std::move(result));
            }

            // Close the channel
            void set_exception(std::exception_ptr const& /*e*/)
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
                return channel_.get(launch::sync, ec);
            }

            // Additional functionality exposed by the channel component
            hpx::future<T> get_generation(std::size_t generation)
            {
                return channel_.get(generation);
            }
            HPX_DEFINE_COMPONENT_ACTION(channel_base, get_generation);

            void set_generation(RemoteType && value, std::size_t generation)
            {
                channel_.set(std::move(value), generation);
            }
            HPX_DEFINE_COMPONENT_ACTION(channel_base, set_generation);

            void close()
            {
                channel_.close();
            }
            HPX_DEFINE_COMPONENT_ACTION(channel_base, close);

        protected:
            lcos::local::channel<result_type> channel_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename RemoteType>
    class channel<T, false, RemoteType>
      : public detail::channel_base<T, RemoteType>
      , public components::component_base<channel<T, false, RemoteType> >
    {
    private:
        typedef detail::channel_base<T, RemoteType> channel_base;
        typedef components::component_base<channel> base_type;

    public:
        channel() = default;

        // disambiguate base classes
        using base_type::finalize;

        typedef typename base_type::wrapping_type wrapping_type;
        typedef typename channel_base::base_type_holder base_type_holder;

        static components::component_type get_component_type()
        {
            return components::get_component_type<channel>();
        }
        static void set_component_type(components::component_type type)
        {
            components::set_component_type<channel>(type);
        }

        using channel_base::get_generation_action;
        using channel_base::set_generation_action;
        using channel_base::close_action;
    };

    template <typename T, typename RemoteType>
    class channel<T, true, RemoteType>
      : public detail::channel_base<T, RemoteType>
      , public components::migration_support<
            components::component_base<channel<T, true, RemoteType> > >
    {
    private:
        typedef detail::channel_base<T, RemoteType> channel_base;
        typedef components::migration_support<
                components::component_base<channel>
            > base_type;

    public:
        channel() = default;

        // disambiguate base classes
        using base_type::finalize;
        typedef typename base_type::wrapping_type wrapping_type;
        typedef typename channel_base::base_type_holder base_type_holder;

        static components::component_type get_component_type()
        {
            return components::get_component_type<channel>();
        }
        static void set_component_type(components::component_type type)
        {
            components::set_component_type<channel>(type);
        }

        using channel_base::get_generation_action;
        using channel_base::set_generation_action;
        using channel_base::close_action;

    private:
        friend class serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & channel_;
        }
    };
}}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_CHANNEL_DECLARATION(...)                                 \
    HPX_REGISTER_CHANNEL_DECLARATION_(__VA_ARGS__)                            \
/**/
#define HPX_REGISTER_CHANNEL_DECLARATION_(...)                                \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_CHANNEL_DECLARATION_, HPX_UTIL_PP_NARG(__VA_ARGS__)      \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_MIGRATABLE_CHANNEL_DECLARATION(...)                      \
    HPX_REGISTER_MIGRATABLE_CHANNEL_DECLARATION_(__VA_ARGS__)                 \
/**/
#define HPX_REGISTER_MIGRATABLE_CHANNEL_DECLARATION_(...)                     \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_MIGRATABLE_CHANNEL_DECLARATION_,                         \
        HPX_UTIL_PP_NARG(__VA_ARGS__)                                         \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_CHANNEL_DECLARATION_1(type)                              \
    HPX_REGISTER_CHANNEL_DECLARATION_3(type, type, false)                     \
/**/
#define HPX_REGISTER_CHANNEL_DECLARATION_2(type, name)                        \
    HPX_REGISTER_CHANNEL_DECLARATION_3(type, name, false)                     \
/**/
#define HPX_REGISTER_CHANNEL_DECLARATION_3(type, name, migratable)            \
    typedef ::hpx::lcos::server::channel< type, migratable>                   \
        BOOST_PP_CAT(__channel_, BOOST_PP_CAT(type, name));                   \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        BOOST_PP_CAT(__channel_, BOOST_PP_CAT(type, name))::get_generation_action,\
        BOOST_PP_CAT(__channel_get_generation_action,                         \
            BOOST_PP_CAT(type, name)));                                       \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        BOOST_PP_CAT(__channel_, BOOST_PP_CAT(type, name))::set_generation_action,\
        BOOST_PP_CAT(__channel_set_generation_action,                         \
            BOOST_PP_CAT(type, name)));                                       \
    HPX_REGISTER_ACTION_DECLARATION(                                          \
        BOOST_PP_CAT(__channel_, BOOST_PP_CAT(type, name))::close_action,     \
        BOOST_PP_CAT(__channel_close_action,                                  \
            BOOST_PP_CAT(type, name)));                                       \
    HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(                             \
        type, type, name, ::hpx::traits::detail::simple_component_tag)        \
/**/

#define HPX_REGISTER_MIGRATABLE_CHANNEL_DECLARATION_1(type)                   \
    HPX_REGISTER_CHANNEL_DECLARATION_3(type, type, true)                      \
/**/
#define HPX_REGISTER_MIGRATABLE_CHANNEL_DECLARATION_2(type, name)             \
    HPX_REGISTER_CHANNEL_DECLARATION_3(type, name, true)                      \
/**/

///////////////////////////////////////////////////////////////////////////////
#define HPX_REGISTER_CHANNEL(...)                                             \
    HPX_REGISTER_CHANNEL_(__VA_ARGS__)                                        \
/**/
#define HPX_REGISTER_CHANNEL_(...)                                            \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_CHANNEL_, HPX_UTIL_PP_NARG(__VA_ARGS__)                  \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_MIGRATABLE_CHANNEL(...)                                  \
    HPX_REGISTER_MIGRATABLE_CHANNEL_(__VA_ARGS__)                             \
/**/
#define HPX_REGISTER_MIGRATABLE_CHANNEL_(...)                                 \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                            \
        HPX_REGISTER_MIGRATABLE_CHANNEL_, HPX_UTIL_PP_NARG(__VA_ARGS__)       \
    )(__VA_ARGS__))                                                           \
/**/

#define HPX_REGISTER_CHANNEL_1(type)                                          \
    HPX_REGISTER_CHANNEL_3(type, type, false)                                 \
/**/
#define HPX_REGISTER_CHANNEL_2(type, name)                                    \
    HPX_REGISTER_CHANNEL_3(type, name, false)                                 \
/**/
#define HPX_REGISTER_CHANNEL_3(type, name, migratable)                        \
    typedef ::hpx::lcos::server::channel< type, migratable>                   \
        BOOST_PP_CAT(__channel_, BOOST_PP_CAT(type, name));                   \
    typedef ::hpx::components::component<                                     \
            BOOST_PP_CAT(__channel_, BOOST_PP_CAT(type, name))                \
        > BOOST_PP_CAT(__channel_component_, name);                           \
    HPX_REGISTER_DERIVED_COMPONENT_FACTORY(                                   \
        BOOST_PP_CAT(__channel_component_, name),                             \
        BOOST_PP_CAT(__channel_component_, name),                             \
        BOOST_PP_STRINGIZE(BOOST_PP_CAT(__base_lco_with_value_channel_, name)));\
    HPX_REGISTER_ACTION(                                                      \
        BOOST_PP_CAT(__channel_, BOOST_PP_CAT(type, name))::get_generation_action,\
        BOOST_PP_CAT(__channel_get_generation_action,                         \
            BOOST_PP_CAT(type, name)));                                       \
    HPX_REGISTER_ACTION(                                                      \
        BOOST_PP_CAT(__channel_, BOOST_PP_CAT(type, name))::set_generation_action,\
        BOOST_PP_CAT(__channel_set_generation_action,                         \
            BOOST_PP_CAT(type, name)));                                       \
    HPX_REGISTER_ACTION(                                                      \
        BOOST_PP_CAT(__channel_, BOOST_PP_CAT(type, name))::close_action,     \
        BOOST_PP_CAT(__channel_close_action,                                  \
            BOOST_PP_CAT(type, name)))                                        \
    HPX_REGISTER_BASE_LCO_WITH_VALUE(                                         \
        type, type, name, ::hpx::traits::detail::simple_component_tag)        \
/**/

#define HPX_REGISTER_MIGRATABLE_CHANNEL_1(type)                               \
    HPX_REGISTER_CHANNEL_3(type, type, true)                                  \
/**/
#define HPX_REGISTER_MIGRATABLE_CHANNEL_2(type, name)                         \
    HPX_REGISTER_CHANNEL_3(type, name, true)                                  \
/**/

#endif
