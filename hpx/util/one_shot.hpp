//  Copyright (c) 2011-2012 Thomas Heller
//  Copyright (c) 2013-2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_ONE_SHOT_HPP
#define HPX_UTIL_ONE_SHOT_HPP

#include <hpx/config.hpp>
#include <hpx/traits/get_function_address.hpp>
#include <hpx/traits/get_function_annotation.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/result_of.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename F>
        class one_shot_wrapper //-V690
        {
        public:
#           if !defined(HPX_DISABLE_ASSERTS)
            // default constructor is needed for serialization
            HPX_CONSTEXPR one_shot_wrapper()
              : _called(false)
            {}

            HPX_CONSTEXPR explicit one_shot_wrapper(F const& f)
              : _f(f)
              , _called(false)
            {}
            HPX_CONSTEXPR explicit one_shot_wrapper(F&& f)
              : _f(std::move(f))
              , _called(false)
            {}

            HPX_CXX14_CONSTEXPR one_shot_wrapper(one_shot_wrapper&& other)
              : _f(std::move(other._f))
              , _called(other._called)
            {
                other._called = true;
            }

            void check_call()
            {
                HPX_ASSERT(!_called);

                _called = true;
            }
#           else
            // default constructor is needed for serialization
            HPX_CONSTEXPR one_shot_wrapper()
            {}

            HPX_CONSTEXPR explicit one_shot_wrapper(F const& f)
              : _f(f)
            {}
            HPX_CONSTEXPR explicit one_shot_wrapper(F&& f)
              : _f(std::move(f))
            {}

            HPX_CONSTEXPR one_shot_wrapper(one_shot_wrapper&& other)
              : _f(std::move(other._f))
            {}

            void check_call()
            {}
#           endif

            template <typename ...Ts>
            HPX_CXX14_CONSTEXPR HPX_HOST_DEVICE
            typename util::invoke_result<F, Ts...>::type
            operator()(Ts&&... vs)
            {
                check_call();

                using invoke_impl = typename detail::dispatch_invoke<F>::type;
                return invoke_impl(std::move(_f))(std::forward<Ts>(vs)...);
            }

            template <typename Archive>
            void serialize(Archive& ar, unsigned int const /*version*/)
            {
                ar & _f;
            }

            std::size_t get_function_address() const
            {
                return traits::get_function_address<F>::call(_f);
            }

            char const* get_function_annotation() const
            {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
                return traits::get_function_annotation<F>::call(_f);
#else
                return nullptr;
#endif
            }

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
            util::itt::string_handle get_function_annotation_itt() const
            {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
                return traits::get_function_annotation_itt<F>::call(_f);
#else
                static util::itt::string_handle sh("one_shot_wrapper");
                return sh;
#endif
            }
#endif

        public: // exposition-only
            F _f;
#           if !defined(HPX_DISABLE_ASSERTS)
            bool _called;
#           endif
        };
    }

    template <typename F>
    HPX_CONSTEXPR detail::one_shot_wrapper<typename std::decay<F>::type>
    one_shot(F&& f)
    {
        typedef
            detail::one_shot_wrapper<typename std::decay<F>::type>
            result_type;

        return result_type(std::forward<F>(f));
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
    template <typename F>
    struct get_function_address<util::detail::one_shot_wrapper<F> >
    {
        static std::size_t
            call(util::detail::one_shot_wrapper<F> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    struct get_function_annotation<util::detail::one_shot_wrapper<F> >
    {
        static char const*
            call(util::detail::one_shot_wrapper<F> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename F>
    struct get_function_annotation_itt<util::detail::one_shot_wrapper<F> >
    {
        static util::itt::string_handle
            call(util::detail::one_shot_wrapper<F> const& f) noexcept
        {
            return f.get_function_annotation_itt();
        }
    };
#endif
#endif
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace serialization
{
    template <typename Archive, typename F>
    void serialize(
        Archive& ar
      , ::hpx::util::detail::one_shot_wrapper<F>& one_shot_wrapper
      , unsigned int const version = 0)
    {
        one_shot_wrapper.serialize(ar, version);
    }
}}

#endif
