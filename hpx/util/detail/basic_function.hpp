//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_BASIC_FUNCTION_HPP
#define HPX_UTIL_DETAIL_BASIC_FUNCTION_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/access.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/util/detail/empty_function.hpp>
#include <hpx/util/detail/function_registration.hpp>
#include <hpx/util/detail/get_table.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>
#include <hpx/util/detail/vtable/serializable_vtable.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/safe_bool.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/identity.hpp>

#include <type_traits>
#include <typeinfo>
#include <utility>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    static bool is_empty_function(F const&, std::false_type) BOOST_NOEXCEPT
    {
        return false;
    }

    template <typename F>
    static bool is_empty_function(F const& f, std::true_type) BOOST_NOEXCEPT
    {
        return f == 0;
    }

    template <typename F>
    static bool is_empty_function(F const& f) BOOST_NOEXCEPT
    {
        std::integral_constant<bool,
            std::is_pointer<F>::value
         || std::is_member_pointer<F>::value
        > is_pointer;
        return is_empty_function(f, is_pointer);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Function>
    struct init_registration;

    template <typename VTablePtr, typename IAr, typename OAr>
    struct serializable_function_vtable_ptr
      : VTablePtr
    {
        char const* name;
        typename serializable_vtable<IAr, OAr>::save_object_t save_object;
        typename serializable_vtable<IAr, OAr>::load_object_t load_object;

        template <typename T>
        serializable_function_vtable_ptr(boost::mpl::identity<T>) BOOST_NOEXCEPT
          : VTablePtr(boost::mpl::identity<T>())
          , name("empty")
          , save_object(&serializable_vtable<IAr, OAr>::template save_object<T>)
          , load_object(&serializable_vtable<IAr, OAr>::template load_object<T>)
        {
            if (!this->empty)
            {
                name = get_function_name<std::pair<
                        serializable_function_vtable_ptr, T
                       > >();
            }
            init_registration<std::pair<
                serializable_function_vtable_ptr, T
            > >::g.register_function();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // registration code for serialization
    template <typename VTablePtr, typename IAr, typename OAr, typename T>
    struct init_registration<
        std::pair<serializable_function_vtable_ptr<VTablePtr, IAr, OAr>, T>
    >
    {
        typedef std::pair<
            serializable_function_vtable_ptr<VTablePtr, IAr, OAr>, T
        > vtable_ptr;

        static automatic_function_registration<vtable_ptr> g;
    };

    template <typename VTablePtr, typename IAr, typename OAr, typename T>
    automatic_function_registration<
        std::pair<serializable_function_vtable_ptr<VTablePtr, IAr, OAr>, T>
    > init_registration<
        std::pair<serializable_function_vtable_ptr<VTablePtr, IAr, OAr>, T>
    >::g =  automatic_function_registration<
                std::pair<serializable_function_vtable_ptr<VTablePtr, IAr, OAr>, T>
            >();

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
                typedef typename std::decay<F>::type target_type;

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
            typedef typename std::decay<T>::type target_type;

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
    template <typename VTablePtr, typename Sig, typename IAr, typename OAr>
    class basic_function;

    template <
        typename VTablePtr, typename R, typename ...Ts
      , typename IAr, typename OAr
    >
    class basic_function<VTablePtr, R(Ts...), IAr, OAr>
      : public function_base<
            serializable_function_vtable_ptr<VTablePtr, IAr, OAr>
          , R(Ts...)
        >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);

        typedef serializable_function_vtable_ptr<VTablePtr, IAr, OAr> vtable_ptr;
        typedef function_base<vtable_ptr, R(Ts...)> base_type;

    public:
        typedef R result_type;

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
                (traits::is_callable<T(Ts...), R>::value)
              , "T shall be Callable with the function signature"
            );

            return base_type::template target<T>();
        }

        template <typename T>
        T* target() const BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                (traits::is_callable<T(Ts...), R>::value)
              , "T shall be Callable with the function signature"
            );

            return base_type::template target<T>();
        }

    private:
        friend class hpx::serialization::access;

        void load(IAr& ar, const unsigned version)
        {
            this->reset();

            bool is_empty = false;
            ar >> is_empty;
            if (!is_empty)
            {
                std::string name;
                ar >> name;

                this->vptr = detail::get_table_ptr<vtable_ptr>(name);
                this->vptr->load_object(&this->object, ar, version);
            }
        }

        void save(OAr& ar, const unsigned version) const
        {
            bool is_empty = this->empty();
            ar << is_empty;
            if (!is_empty)
            {
                std::string function_name = this->vptr->name;
                ar << function_name;

                this->vptr->save_object(&this->object, ar, version);
            }
        }

        HPX_SERIALIZATION_SPLIT_MEMBER()
    };

    template <typename VTablePtr, typename R, typename ...Ts>
    class basic_function<VTablePtr, R(Ts...), void, void>
      : public function_base<VTablePtr, R(Ts...)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);

        typedef function_base<VTablePtr, R(Ts...)> base_type;

    public:
        typedef R result_type;

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
            static_assert(
                (traits::is_callable<T(Ts...), R>::value)
              , "T shall be Callable with the function signature"
            );

            return base_type::template target<T>();
        }

        template <typename T>
        T* target() const BOOST_NOEXCEPT
        {
            static_assert(
                (traits::is_callable<T(Ts...), R>::value)
              , "T shall be Callable with the function signature"
            );

            return base_type::template target<T>();
        }
    };

    template <typename Sig, typename VTablePtr, typename IAr, typename OAr>
    static bool is_empty_function(
        basic_function<VTablePtr, Sig, IAr, OAr> const& f) BOOST_NOEXCEPT
    {
        return f.empty();
    }
}}}

// Pseudo registration for empty functions.
// We don't want to serialize empty functions.
namespace hpx { namespace util { namespace detail
{
    template <typename VTablePtr, typename Sig>
    struct get_function_name_impl<
        std::pair<
            hpx::util::detail::serializable_function_vtable_ptr<
                VTablePtr
              , hpx::serialization::input_archive
              , hpx::serialization::output_archive
            >
          , hpx::util::detail::empty_function<Sig>
        >
    >
    {
        static char const * call()
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "get_function_name<empty_function>");
            return "";
        }
    };
}}}

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename VTablePtr, typename Sig>
    struct needs_automatic_registration<
        std::pair<
            hpx::util::detail::serializable_function_vtable_ptr<
                VTablePtr
              , hpx::serialization::input_archive
              , hpx::serialization::output_archive
            >
          , hpx::util::detail::empty_function<Sig>
        >
    > : boost::mpl::false_
    {};
}}

#endif
