//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_FUNCTION_HPP
#define HPX_UTIL_FUNCTION_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/traits/get_function_address.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/util/detail/basic_function.hpp>
#include <hpx/util/detail/function_registration.hpp>
#include <hpx/util/detail/vtable/function_vtable.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>
#include <hpx/util_fwd.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig, bool Serializable>
    class function;

    template <typename R, typename ...Ts, bool Serializable>
    class function<R(Ts...), Serializable>
      : public detail::basic_function<
            detail::function_vtable<R(Ts...)>
          , R(Ts...), Serializable
        >
    {
        typedef detail::function_vtable<R(Ts...)> vtable;
        typedef detail::basic_function<vtable, R(Ts...), Serializable> base_type;

    public:
        typedef typename base_type::result_type result_type;

        function() HPX_NOEXCEPT
          : base_type()
        {}

        function(std::nullptr_t) HPX_NOEXCEPT
          : base_type()
        {}

        function(function const& other)
          : base_type()
        {
            detail::vtable::_delete<
                detail::empty_function<R(Ts...)>
            >(this->object);

            this->vptr = other.vptr;
            if (!this->vptr->empty)
            {
                this->vptr->copy(this->object, other.object);
            }
        }

        function(function&& other) HPX_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}

        template <typename F, typename FD = typename std::decay<F>::type,
            typename Enable = typename std::enable_if<
                !std::is_same<FD, function>::value
             && traits::is_callable<FD&(Ts...), R>::value
            >::type>
        function(F&& f)
          : base_type()
        {
            static_assert(
                std::is_constructible<FD, FD const&>::value,
                "F shall be CopyConstructible");
            assign(std::forward<F>(f));
        }

        function& operator=(function const& other)
        {
            if (this != &other)
            {
                reset();
                detail::vtable::_delete<
                    detail::empty_function<R(Ts...)>
                >(this->object);

                this->vptr = other.vptr;
                if (!this->vptr->empty)
                {
                    this->vptr->copy(this->object, other.object);
                }
            }
            return *this;
        }

        function& operator=(function&& other) HPX_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }

        template <typename F, typename FD = typename std::decay<F>::type,
            typename Enable = typename std::enable_if<
                !std::is_same<FD, function>::value
             && traits::is_callable<FD&(Ts...), R>::value
            >::type>
        function& operator=(F&& f)
        {
            static_assert(
                std::is_constructible<FD, FD const&>::value,
                "F shall be CopyConstructible");
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
        function<Sig, Serializable> const& f) HPX_NOEXCEPT
    {
        return f.empty();
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    template <typename Sig, bool Serializable>
    struct get_function_address<util::function<Sig, Serializable> >
    {
        static std::size_t
            call(util::function<Sig, Serializable> const& f) HPX_NOEXCEPT
        {
            return f.get_function_address();
        }
    };
}}

///////////////////////////////////////////////////////////////////////////////
#define HPX_UTIL_REGISTER_FUNCTION_DECLARATION(Sig, Functor, Name)            \
    HPX_DECLARE_GET_FUNCTION_NAME(function_vtable<Sig>, Functor, Name)        \
/**/

#define HPX_UTIL_REGISTER_FUNCTION(Sig, Functor, Name)                        \
    HPX_DEFINE_GET_FUNCTION_NAME(function_vtable<Sig>, Functor, Name)         \
/**/

#endif
