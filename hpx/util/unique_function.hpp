//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_UNIQUE_FUNCTION_HPP
#define HPX_UTIL_UNIQUE_FUNCTION_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/traits/get_function_address.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/util/detail/basic_function.hpp>
#include <hpx/util/detail/function_registration.hpp>
#include <hpx/util/detail/vtable/unique_function_vtable.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>
#include <hpx/util_fwd.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig, bool Serializable>
    class unique_function;

    template <typename R, typename ...Ts, bool Serializable>
    class unique_function<R(Ts...), Serializable>
      : public detail::basic_function<
            detail::unique_function_vtable<R(Ts...)>
          , R(Ts...), Serializable
        >
    {
        typedef detail::unique_function_vtable<R(Ts...)> vtable;
        typedef detail::basic_function<vtable, R(Ts...), Serializable> base_type;

        HPX_MOVABLE_ONLY(unique_function);

    public:
        typedef typename base_type::result_type result_type;

        unique_function() HPX_NOEXCEPT
          : base_type()
        {}

        unique_function(std::nullptr_t) HPX_NOEXCEPT
          : base_type()
        {}

        unique_function(unique_function&& other) HPX_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}

        template <typename F, typename FD = typename std::decay<F>::type,
            typename Enable = typename std::enable_if<
                !std::is_same<FD, unique_function>::value
             && traits::is_callable<FD&(Ts...), R>::value
            >::type>
        unique_function(F&& f)
          : base_type()
        {
            assign(std::forward<F>(f));
        }

        unique_function& operator=(unique_function&& other) HPX_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }

        template <typename F, typename FD = typename std::decay<F>::type,
            typename Enable = typename std::enable_if<
                !std::is_same<FD, unique_function>::value
             && traits::is_callable<FD&(Ts...), R>::value
            >::type>
        unique_function& operator=(F&& f)
        {
            assign(std::forward<F>(f));
            return *this;
        }

        using base_type::operator();
        using base_type::assign;
        using base_type::reset;
        using base_type::empty;
        using base_type::target_type;
        using base_type::target;
    };

    template <typename Sig, bool Serializable>
    static bool is_empty_function(
        unique_function<Sig, Serializable> const& f) HPX_NOEXCEPT
    {
        return f.empty();
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    template <typename Sig, bool Serializable>
    struct get_function_address<util::unique_function<Sig, Serializable> >
    {
        static std::size_t
            call(util::unique_function<Sig, Serializable> const& f) HPX_NOEXCEPT
        {
            return f.get_function_address();
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_UTIL_REGISTER_UNIQUE_FUNCTION_DECLARATION(Sig, Functor, Name)     \
    HPX_DECLARE_GET_FUNCTION_NAME(unique_function_vtable<Sig>, Functor, Name) \
/**/

#define HPX_UTIL_REGISTER_UNIQUE_FUNCTION(Sig, Functor, Name)                 \
    HPX_DEFINE_GET_FUNCTION_NAME(unique_function_vtable<Sig>, Functor, Name)  \
/**/

#endif
