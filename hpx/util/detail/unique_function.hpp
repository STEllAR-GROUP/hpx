//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_UTIL_DETAIL_UNIQUE_FUNCTION_HPP
#define HPX_UTIL_DETAIL_UNIQUE_FUNCTION_HPP

#include <hpx/config/forceinline.hpp>

#include <hpx/util/detail/function_template.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/detail/vtable_ptr_base_fwd.hpp>
#include <hpx/util/detail/vtable_ptr_fwd.hpp>
#include <hpx/util/detail/vtable.hpp>
#include <hpx/util/detail/serialization_registration.hpp>
#include <hpx/util/safe_bool.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/serialize_empty_type.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/polymorphic_factory.hpp>

#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>

#include <boost/serialization/utility.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/tracking.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/is_member_pointer.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/or.hpp>

#include <stdexcept>

#ifndef HPX_FUNCTION_VERSION
#define HPX_FUNCTION_VERSION 0x10
#endif

namespace hpx { namespace util { namespace detail
{
    template <
        typename Sig
      , typename IArchive = void, typename OArchive = void
    >
    struct unique_function;

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Sig
      , typename IArchive = void, typename OArchive = void
    >
    struct unique_function_base;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig, typename IArchive, typename OArchive>
    bool is_empty_function(unique_function<Sig, IArchive, OArchive> const& f) BOOST_NOEXCEPT
    {
        return f.empty();
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Function>
    struct invalidate_function
    {
        explicit invalidate_function(Function& f)
          : f_(f)
        {}

        ~invalidate_function()
        {
            f_.reset();
        }

        Function& f_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig, typename IArchive, typename OArchive>
    struct unique_function : unique_function_base<Sig, IArchive, OArchive>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function);

    public:
        typedef unique_function_base<Sig, IArchive, OArchive> base_type;
        typedef typename base_type::result_type result_type;

        unique_function() BOOST_NOEXCEPT
          : base_type() {}

        template <typename Functor>
        unique_function(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function
                  , typename util::decay<Functor>::type
                >::type
            >::type * = 0
        ) : base_type(std::forward<Functor>(f))
        {}

        unique_function(unique_function && other) BOOST_NOEXCEPT
          : base_type(std::move(static_cast<base_type &&>(other)))
        {}

        unique_function& operator=(unique_function && t) BOOST_NOEXCEPT
        {
            this->base_type::operator=(std::move(static_cast<base_type &&>(t)));
            return *this;
        }

    private:
        friend class boost::serialization::access;

        void load(IArchive &ar, const unsigned version)
        {
            bool is_empty;
            ar.load(is_empty);

            if (is_empty)
            {
                this->reset();
            }
            else
            {
                typedef
                    typename base_type::vtable_virtbase_type
                    vtable_virtbase_type;

                typedef
                    typename base_type::vtable_ptr_type
                    vtable_ptr_type;

                std::string function_name;
                ar.load(function_name);

                boost::shared_ptr<vtable_virtbase_type> p(
                    util::polymorphic_factory<
                        vtable_virtbase_type
                    >::create(function_name));

                this->vptr = static_cast<vtable_ptr_type*>(p->get_ptr());
                this->vptr->load_object(&this->object, ar, version);
            }
        }

        void save(OArchive &ar, const unsigned version) const
        {
            bool is_empty = this->empty();
            ar.save(is_empty);

            if (!is_empty)
            {
                std::string function_name = this->vptr->get_function_name();
                ar.save(function_name);

                this->vptr->save_object(&this->object, ar, version);
            }
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER()
    };

    template <typename Sig>
    struct unique_function<Sig, void, void>
      : unique_function_base<Sig, void, void>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function);

    public:
        typedef unique_function_base<Sig, void, void> base_type;
        typedef typename base_type::result_type result_type;

        unique_function() BOOST_NOEXCEPT
          : base_type() {}

        template <typename Functor>
        unique_function(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function
                  , typename util::decay<Functor>::type
                >::type
            >::type * = 0
        ) : base_type(std::forward<Functor>(f))
        {}

        unique_function(unique_function && other) BOOST_NOEXCEPT
          : base_type(std::move(static_cast<base_type &&>(other)))
        {}

        unique_function& operator=(unique_function && t) BOOST_NOEXCEPT
        {
            this->base_type::operator=(std::move(static_cast<base_type &&>(t)));
            return *this;
        }
    };
}}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/util/detail/preprocessed/unique_function.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/unique_function_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            0                                                                   \
          , HPX_FUNCTION_ARGUMENT_LIMIT                                         \
          , <hpx/util/detail/unique_function.hpp>                               \
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
    template <
        typename R
      BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
      , typename IArchive, typename OArchive
    >
    struct unique_function_base<
        R(BOOST_PP_ENUM_PARAMS(N, A))
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);

    public:
        typedef R result_type;

        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;

        typedef
            detail::vtable_ptr_base<
                R(BOOST_PP_ENUM_PARAMS(N, A))
              , IArchive, OArchive
            > vtable_ptr_type;

        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}

        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }

        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * /*dummy*/ = 0
        ) : vptr(get_empty_table_ptr())
          , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename util::decay<Functor>::type
                    functor_type;

                vptr = get_table_ptr<functor_type>();
                if (sizeof(functor_type) <= sizeof(void *))  // is_small
                {
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
        }

        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }

        template <typename Functor>
        unique_function_base & assign(Functor && f)
        {
            if (this == &f)
                return *this;

            typedef
                typename util::decay<Functor>::type
                functor_type;

            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if(vptr == f_vptr && !empty())
            {
                if (sizeof(functor_type) <= sizeof(void *))  // is_small
                {
                    vptr->destruct(&object);
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *))  // is_small
                    {
                        new (&object) functor_type(std::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(std::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }

        template <typename T>
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }

        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
        {
            if(this != &t)
            {
                reset();
                vptr = t.vptr;
                object = t.object;
                t.vptr = get_empty_table_ptr();
                t.object = 0;
            }

            return *this;
        }

        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }

        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }

        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
        }

        bool operator!() const BOOST_NOEXCEPT
        {
            return empty();
        }

        void reset() BOOST_NOEXCEPT
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }

        static vtable_ptr_type* get_empty_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_empty_table<
                        R(BOOST_PP_ENUM_PARAMS(N, A))
                    >::template get<IArchive, OArchive>();
        }

        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R(BOOST_PP_ENUM_PARAMS(N, A))
                    >::template get<true, IArchive, OArchive>();
        }

        BOOST_FORCEINLINE R operator()(BOOST_PP_ENUM_BINARY_PARAMS(N, A, a))
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                BOOST_PP_COMMA_IF(N) HPX_ENUM_FORWARD_ARGS(N, A, a));
        }

    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}}

#undef N
#endif
