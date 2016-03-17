//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013-2016 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_UNIQUE_FUNCTION_TEMPLATE_HPP
#define HPX_UTIL_DETAIL_UNIQUE_FUNCTION_TEMPLATE_HPP

#include <hpx/config.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/traits/get_function_address.hpp>
#include <hpx/util/detail/basic_function.hpp>
#include <hpx/util/detail/vtable/callable_vtable.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    template <typename Sig>
    struct unique_function_vtable_ptr
    {
        typename callable_vtable<Sig>::invoke_t invoke;
        vtable::get_type_t get_type;
        vtable::destruct_t destruct;
        vtable::delete_t delete_;
        vtable::get_function_address_t get_function_address;
        bool empty;

        template <typename T>
        unique_function_vtable_ptr(construct_vtable<T>) HPX_NOEXCEPT
          : invoke(&callable_vtable<Sig>::template invoke<T>)
          , get_type(&vtable::template get_type<T>)
          , destruct(&vtable::template destruct<T>)
          , delete_(&vtable::template delete_<T>)
          , get_function_address(&vtable::template get_function_address<T>)
          , empty(std::is_same<T, empty_function<Sig> >::value)
        {}

        template <typename T, typename Arg>
        HPX_FORCEINLINE static void construct(void** v, Arg&& arg)
        {
            vtable::construct<T>(v, std::forward<Arg>(arg));
        }

        template <typename T, typename Arg>
        HPX_FORCEINLINE static void reconstruct(void** v, Arg&& arg)
        {
            vtable::reconstruct<T>(v, std::forward<Arg>(arg));
        }
    };
}}}

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig, bool Serializable = true>
    class unique_function;

    template <typename R, typename ...Ts, bool Serializable>
    class unique_function<R(Ts...), Serializable>
      : public detail::basic_function<
            detail::unique_function_vtable_ptr<R(Ts...)>
          , R(Ts...), Serializable
        >
    {
        typedef detail::unique_function_vtable_ptr<R(Ts...)> vtable_ptr;
        typedef detail::basic_function<vtable_ptr, R(Ts...), Serializable> base_type;

        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function)

    public:
        typedef typename base_type::result_type result_type;

        unique_function() HPX_NOEXCEPT
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

    ///////////////////////////////////////////////////////////////////////////
#   ifdef HPX_HAVE_CXX11_ALIAS_TEMPLATES

    template <typename Sig>
    using unique_function_nonser = unique_function<Sig, false>;

#   else

    template <typename T>
    class unique_function_nonser;

    template <typename R, typename ...Ts>
    class unique_function_nonser<R(Ts...)>
      : public unique_function<R(Ts...), false>
    {
        typedef unique_function<R(Ts...), false> base_type;

        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_nonser);

    public:
        unique_function_nonser() HPX_NOEXCEPT
          : base_type()
        {}

        unique_function_nonser(unique_function_nonser&& other) HPX_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}

        template <typename F, typename FD = typename std::decay<F>::type,
            typename Enable = typename std::enable_if<
                !std::is_same<FD, unique_function_nonser>::value
             && traits::is_callable<FD&(Ts...), R>::value
            >::type>
        unique_function_nonser(F&& f)
          : base_type(std::forward<F>(f))
        {}

        unique_function_nonser& operator=(unique_function_nonser&& other) HPX_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }

        template <typename F, typename FD = typename std::decay<F>::type,
            typename Enable = typename std::enable_if<
                !std::is_same<FD, unique_function_nonser>::value
             && traits::is_callable<FD&(Ts...), R>::value
            >::type>
        unique_function_nonser& operator=(F&& f)
        {
            base_type::operator=(std::forward<F>(f));
            return *this;
        }
    };

    template <typename Sig>
    static bool is_empty_function(
        unique_function_nonser<Sig> const& f) HPX_NOEXCEPT
    {
        return f.empty();
    }

#   endif /*HPX_HAVE_CXX11_ALIAS_TEMPLATES*/
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

#   ifndef HPX_HAVE_CXX11_ALIAS_TEMPLATES
    template <typename Sig>
    struct get_function_address<util::unique_function_nonser<Sig> >
    {
        static std::size_t
            call(util::unique_function_nonser<Sig> const& f) HPX_NOEXCEPT
        {
            return f.get_function_address();
        }
    };
#   endif /*HPX_HAVE_CXX11_ALIAS_TEMPLATES*/
}}

#endif
