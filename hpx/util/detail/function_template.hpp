//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_FUNCTION_TEMPLATE_HPP
#define HPX_UTIL_DETAIL_FUNCTION_TEMPLATE_HPP

#include <hpx/config.hpp>
#include <hpx/util/detail/basic_function.hpp>
#include <hpx/util/detail/vtable/callable_vtable.hpp>
#include <hpx/util/detail/vtable/copyable_vtable.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>

#include <boost/mpl/identity.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig>
    struct function_vtable_ptr
    {
        typename callable_vtable<Sig>::invoke_t invoke;
        copyable_vtable::copy_t copy;
        vtable::get_type_t get_type;
        vtable::destruct_t destruct;
        vtable::delete_t delete_;
        bool empty;

        template <typename T>
        function_vtable_ptr(boost::mpl::identity<T>) BOOST_NOEXCEPT
          : invoke(&callable_vtable<Sig>::template invoke<T>)
          , copy(&copyable_vtable::template copy<T>)
          , get_type(&vtable::template get_type<T>)
          , destruct(&vtable::template destruct<T>)
          , delete_(&vtable::template delete_<T>)
          , empty(std::is_same<T, empty_function<Sig> >::value)
        {}

        template <typename T, typename Arg>
        BOOST_FORCEINLINE static void construct(void** v, Arg&& arg)
        {
            vtable::construct<T>(v, std::forward<Arg>(arg));
        }

        template <typename T, typename Arg>
        BOOST_FORCEINLINE static void reconstruct(void** v, Arg&& arg)
        {
            vtable::reconstruct<T>(v, std::forward<Arg>(arg));
        }
    };
}}}

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Sig
      , typename IAr = serialization::input_archive
      , typename OAr = serialization::output_archive
    >
    class function
      : public detail::basic_function<
            detail::function_vtable_ptr<Sig>
          , Sig, IAr, OAr
        >
    {
        typedef detail::function_vtable_ptr<Sig> vtable_ptr;
        typedef detail::basic_function<vtable_ptr, Sig, IAr, OAr> base_type;

    public:
        typedef typename base_type::result_type result_type;

        function() BOOST_NOEXCEPT
          : base_type()
        {}

        function(function const& other)
          : base_type()
        {
            detail::vtable::destruct<detail::empty_function<Sig> >(&this->object);

            this->vptr = other.vptr;
            if (!this->vptr->empty)
            {
                this->vptr->copy(&this->object, &other.object);
            }
        }

        function(function&& other) BOOST_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}

        template <typename F, typename Enable =
            typename std::enable_if<
                !std::is_same<F, function>::value
            >::type>
        function(F&& f)
          : base_type()
        {
            static_assert(
                std::is_copy_constructible<typename decay<F>::type>::value,
                "F shall be CopyConstructible");
            assign(std::forward<F>(f));
        }

        function& operator=(function const& other)
        {
            if (this != &other)
            {
                reset();
                detail::vtable::destruct<detail::empty_function<Sig> >(&this->object);

                this->vptr = other.vptr;
                if (!this->vptr->empty)
                {
                    this->vptr->copy(&this->object, &other.object);
                }
            }
            return *this;
        }

        function& operator=(function&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }

        template <typename F, typename Enable =
            typename std::enable_if<
                !std::is_same<F, function>::value
            >::type>
        function& operator=(F&& f)
        {
            static_assert(
                std::is_copy_constructible<typename decay<F>::type>::value,
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

    template <typename Sig, typename IAr, typename OAr>
    static bool is_empty_function(function<Sig, IAr,
        OAr> const& f) BOOST_NOEXCEPT
    {
        return f.empty();
    }

    ///////////////////////////////////////////////////////////////////////////
#   ifdef HPX_HAVE_CXX11_ALIAS_TEMPLATES

    template <typename Sig>
    using function_nonser = function<Sig, void, void>;

#   else

    template <typename Sig>
    class function_nonser
      : public function<Sig, void, void>
    {
        typedef function<Sig, void, void> base_type;

    public:
        function_nonser() BOOST_NOEXCEPT
          : base_type()
        {}

        function_nonser(function_nonser const& other)
          : base_type(static_cast<base_type const&>(other))
        {}

        function_nonser(function_nonser&& other) BOOST_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}

        template <typename F, typename Enable =
            typename std::enable_if<
                !std::is_same<F, function_nonser>::value
            >::type>
        function_nonser(F&& f)
          : base_type(std::forward<F>(f))
        {}

        function_nonser& operator=(function_nonser const& other)
        {
            base_type::operator=(static_cast<base_type const&>(other));
            return *this;
        }

        function_nonser& operator=(function_nonser&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }

        template <typename F, typename Enable =
            typename std::enable_if<
                !std::is_same<F, function_nonser>::value
            >::type>
        function_nonser& operator=(F&& f)
        {
            base_type::operator=(std::forward<F>(f));
            return *this;
        }
    };

    template <typename Sig>
    static bool is_empty_function(function_nonser<Sig> const& f) BOOST_NOEXCEPT
    {
        return f.empty();
    }

#   endif /*HPX_HAVE_CXX11_ALIAS_TEMPLATES*/
}}

#endif
