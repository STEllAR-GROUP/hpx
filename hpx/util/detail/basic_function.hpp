//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_UTIL_DETAIL_BASIC_FUNCTION_HPP
#define HPX_UTIL_DETAIL_BASIC_FUNCTION_HPP

#include <hpx/config.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/util/detail/empty_function.hpp>
#include <hpx/util/detail/get_table.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/safe_bool.hpp>

#include <boost/static_assert.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/is_member_pointer.hpp>

#include <typeinfo>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    static bool is_empty_function(F const&, boost::mpl::false_) BOOST_NOEXCEPT
    {
        return false;
    }

    template <typename F>
    static bool is_empty_function(F const& f, boost::mpl::true_) BOOST_NOEXCEPT
    {
        return f == 0;
    }

    template <typename F>
    static bool is_empty_function(F const& f) BOOST_NOEXCEPT
    {
        boost::mpl::bool_<
            boost::is_pointer<F>::value
         || boost::is_member_pointer<F>::value
        > is_pointer;
        return is_empty_function(f, is_pointer);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename VTablePtr, typename Sig>
    class function_base
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(function_base);

    public:
        function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {
            vtable::default_construct<empty_function<Sig> >(&object);
        }

        function_base(function_base&& other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object) // move-construct
        {
            other.vptr = get_empty_table_ptr();
            vtable::default_construct<empty_function<Sig> >(&other.object);
        }

        ~function_base()
        {
            vptr->delete_(&object);
        }

        function_base& operator=(function_base&& other) BOOST_NOEXCEPT
        {
            if (this != &other)
            {
                swap(other);
                other.reset();
            }
            return *this;
        }

        template <typename F>
        void assign(F&& f)
        {
            if (!is_empty_function(f))
            {
                typedef typename util::decay<F>::type target_type;

                VTablePtr const* f_vptr = get_table_ptr<target_type>();
                if (vptr == f_vptr)
                {
                    vtable::reconstruct<target_type>(&object, std::forward<F>(f));
                } else {
                    reset();
                    vtable::destruct<empty_function<Sig> >(&this->object);

                    vptr = f_vptr;
                    vtable::construct<target_type>(&object, std::forward<F>(f));
                }
            } else {
                reset();
            }
        }

        void reset() BOOST_NOEXCEPT
        {
            if (!vptr->empty)
            {
                vptr->delete_(&object);

                vptr = get_empty_table_ptr();
                vtable::default_construct<empty_function<Sig> >(&object);
            }
        }

        void swap(function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object); // swap
        }

        bool empty() const BOOST_NOEXCEPT
        {
            return vptr->empty;
        }

#       ifndef BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
        explicit operator bool() const BOOST_NOEXCEPT
        {
            return !empty();
        }
#       else
        operator typename util::safe_bool<function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<function_base>()(!empty());
        }
#       endif

        std::type_info const& target_type() const BOOST_NOEXCEPT
        {
            return empty() ? typeid(void) : vptr->get_type();
        }

        template <typename T>
        T* target() const BOOST_NOEXCEPT
        {
            typedef typename util::decay<T>::type target_type;

            VTablePtr const* f_vptr = get_table_ptr<target_type>();
            if (vptr != f_vptr || empty())
                return 0;

            return &vtable::get<target_type>(&object);
        }

    private:
        static VTablePtr const* get_empty_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<VTablePtr, detail::empty_function<Sig> >();
        }

        template <typename T>
        static VTablePtr const* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<VTablePtr, T>();
        }

    protected:
        VTablePtr const *vptr;
        mutable void *object;
    };

    template <typename Sig, typename VTablePtr>
    static bool is_empty_function(function_base<VTablePtr, Sig> const& f) BOOST_NOEXCEPT
    {
        return f.empty();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename VTablePtr, typename Sig>
    class basic_function;

    template <typename VTablePtr, typename R>
    class basic_function<VTablePtr, R()>
      : public function_base<VTablePtr, R()>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);

        typedef function_base<VTablePtr, R()> base_type;

    public:
        typedef R result_type;

        template <typename T>
        struct is_callable
          : traits::is_callable<T()>
        {};

        basic_function() BOOST_NOEXCEPT
          : base_type()
        {}

        basic_function(basic_function&& other) BOOST_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}

        basic_function& operator=(basic_function&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }

        BOOST_FORCEINLINE R operator()() const
        {
            return this->vptr->invoke(&this->object);
        }

        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );

            return base_type::template target<T>();
        }

        template <typename T>
        T* target() const BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );

            return base_type::template target<T>();
        }
    };

    template <typename Sig, typename VTablePtr>
    static bool is_empty_function(basic_function<VTablePtr, Sig> const& f) BOOST_NOEXCEPT
    {
        return f.empty();
    }
}}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/util/detail/preprocessed/basic_function.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/basic_function_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            1                                                                   \
          , HPX_FUNCTION_ARGUMENT_LIMIT                                         \
          , <hpx/util/detail/basic_function.hpp>                                \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#endif

#else

#define N BOOST_PP_ITERATION()

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename VTablePtr,
        typename R, BOOST_PP_ENUM_PARAMS(N, typename A)>
    class basic_function<VTablePtr, R(BOOST_PP_ENUM_PARAMS(N, A))>
      : public function_base<VTablePtr, R(BOOST_PP_ENUM_PARAMS(N, A))>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);

        typedef function_base<VTablePtr, R(BOOST_PP_ENUM_PARAMS(N, A))> base_type;

    public:
        typedef R result_type;

        template <typename T>
        struct is_callable
          : traits::is_callable<T(BOOST_PP_ENUM_PARAMS(N, A))>
        {};

        basic_function() BOOST_NOEXCEPT
          : base_type()
        {}

        basic_function(basic_function&& other) BOOST_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}

        basic_function& operator=(basic_function&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }

        BOOST_FORCEINLINE R operator()(BOOST_PP_ENUM_BINARY_PARAMS(N, A, a)) const
        {
            return this->vptr->invoke(&this->object,
                HPX_ENUM_FORWARD_ARGS(N, A, a));
        }

        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );

            return base_type::template target<T>();
        }

        template <typename T>
        T* target() const BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );

            return base_type::template target<T>();
        }
    };
}}}

#undef N

#endif
