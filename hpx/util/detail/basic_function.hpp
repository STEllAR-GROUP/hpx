//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

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

        static VTablePtr const empty_table;

    public:
        function_base() BOOST_NOEXCEPT
          : vptr(&empty_table)
          , object(0)
        {
            vtable::default_construct<empty_function<Sig> >(&object);
        }

        function_base(function_base&& other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object) // move-construct
        {
            other.vptr = &empty_table;
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

                vptr = &empty_table;
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

#       ifdef HPX_HAVE_CXX11_EXPLICIT_CONVERSION_OPERATORS
        explicit operator bool() const BOOST_NOEXCEPT
        {
            return !empty();
        }
#       else
        operator typename util::safe_bool<function_base>
            ::result_type() const BOOST_NOEXCEPT
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
        template <typename T>
        static VTablePtr const* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<VTablePtr, T>();
        }

    protected:
        VTablePtr const *vptr;
        mutable void *object;
    };

    template <typename VTablePtr, typename Sig>
    VTablePtr const function_base<VTablePtr, Sig>::empty_table =
        boost::mpl::identity<detail::empty_function<Sig> >();

    template <typename Sig, typename VTablePtr>
    static bool is_empty_function(function_base<VTablePtr, Sig> const& f) BOOST_NOEXCEPT
    {
        return f.empty();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename VTablePtr, typename Sig>
    class basic_function;

    template <typename VTablePtr, typename R, typename ...Ts>
    class basic_function<VTablePtr, R(Ts...)>
      : public function_base<VTablePtr, R(Ts...)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);

        typedef function_base<VTablePtr, R(Ts...)> base_type;

    public:
        typedef R result_type;

        template <typename T>
        struct is_callable
          : traits::is_callable<T(Ts...)>
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

        BOOST_FORCEINLINE R operator()(Ts... vs) const
        {
            return this->vptr->invoke(&this->object, std::forward<Ts>(vs)...);
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

#endif
