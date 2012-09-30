//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#if !BOOST_PP_IS_ITERATING

#ifndef HPX_UTIL_FUNCTION_TEMPLATE_HPP
#define HPX_UTIL_FUNCTION_TEMPLATE_HPP

#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#include <boost/serialization/utility.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/type_traits/decay.hpp>
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <hpx/util/detail/remove_reference.hpp>
#include <hpx/util/detail/vtable_ptr_base_fwd.hpp>
#include <hpx/util/detail/vtable_ptr_fwd.hpp>
#include <hpx/util/detail/serialization_registration.hpp>
#include <hpx/util/safe_bool.hpp>
#include <hpx/util/move.hpp>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/is_member_pointer.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/or.hpp>

#ifndef HPX_FUNCTION_VERSION
#define HPX_FUNCTION_VERSION 0x10
#endif

namespace hpx { namespace util
{
    namespace detail
    {
        template <
            typename Functor
          , typename Sig
        >
        struct get_table;

        template <bool>
        struct vtable;

        ///////////////////////////////////////////////////////////////////////
        template <typename Functor>
        struct is_empty_function_impl
        {
            // in the general case the functor is not empty
            static bool call(Functor const&, boost::mpl::false_)
            {
                return false;
            }

            static bool call(Functor const& f, boost::mpl::true_)
            {
                return f == 0;
            }

            static bool call(Functor const& f)
            {
                return call(f
                    , boost::mpl::or_<
                          boost::is_pointer<Functor>
                        , boost::is_member_pointer<Functor>
                      >()
                );
            }
        };

        template <typename Functor>
        bool is_empty_function(Functor const& f)
        {
            return is_empty_function_impl<Functor>::call(f);
        }
    }

    template <
        typename Sig
      , typename IArchive = void
      , typename OArchive = void
    >
    struct function_base;

    template <
        typename Sig
      , typename IArchive = portable_binary_iarchive
      , typename OArchive = portable_binary_oarchive
    >
    struct function : function_base<Sig, IArchive, OArchive>
    {
        using function_base<Sig, IArchive, OArchive>::reset;

        typedef function_base<Sig, IArchive, OArchive> base_type;
        function() : base_type() {}

        template <typename Functor>
        function(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * = 0
        )
            : base_type(boost::forward<Functor>(f))
        {}

        function(function const & other)
            : base_type(static_cast<base_type const &>(other))
        {}

        function(BOOST_RV_REF(function) other)
            : base_type(boost::move(static_cast<BOOST_RV_REF(base_type)>(other)))
        {}

        function& operator=(BOOST_COPY_ASSIGN_REF(function) t)
        {
            this->base_type::operator=(t);
            return *this;
        }

        function& operator=(BOOST_RV_REF(function) t)
        {
            this->base_type::operator=(boost::move(static_cast<BOOST_RV_REF(base_type)>(t)));
            return *this;
        }

        void clear() { reset(); }

    private:
        BOOST_COPYABLE_AND_MOVABLE(function)

        friend class boost::serialization::access;

        void load(IArchive &ar, const unsigned version)
        {
            bool is_empty;
            ar & is_empty;

            if(is_empty)
            {
                this->reset();
            }
            else
            {
                typename base_type::vtable_ptr_type *p = 0;
                ar >> p;
                this->vptr = p->get_ptr();
                delete p;
                this->vptr->load_object(&this->object, ar, version);
            }
        }

        void save(OArchive &ar, const unsigned version) const
        {
            bool is_empty = this->empty();
            ar & is_empty;
            if(!this->empty())
            {
                ar << this->vptr;
                this->vptr->save_object(&this->object, ar, version);
            }
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER()

    };

    template <
        typename Sig
    >
    struct function<Sig, void, void> : function_base<Sig, void, void>
    {
        using function_base<Sig, void, void>::reset;

        typedef function_base<Sig, void, void> base_type;
        function() : base_type() {}

        template <typename Functor>
        function(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * = 0
        )
            : base_type(boost::forward<Functor>(f))
        {}

        function(function const & other)
            : base_type(static_cast<BOOST_COPY_ASSIGN_REF(base_type)>(other))
        {}

        function(BOOST_RV_REF(function) other)
            : base_type(boost::move(static_cast<BOOST_RV_REF(base_type)>(other)))
        {}

        function& operator=(BOOST_COPY_ASSIGN_REF(function) t)
        {
            this->base_type::operator=(t);
            return *this;
        }

        function& operator=(BOOST_RV_REF(function) t)
        {
            this->base_type::operator=(boost::move(static_cast<BOOST_RV_REF(base_type)>(t)));
            return *this;
        }

        void clear() { reset(); }

    private:
        BOOST_COPYABLE_AND_MOVABLE(function)
    };


    template <
        typename Sig
    >
    struct function_nonser : function_base<Sig, void, void>
    {
        using function_base<Sig, void, void>::reset;

        typedef function_base<Sig, void, void> base_type;
        function_nonser() : base_type() {}

        template <typename Functor>
        function_nonser(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_nonser
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * = 0
        )
            : base_type(boost::forward<Functor>(f))
        {}

        function_nonser(function_nonser const & other)
            : base_type(static_cast<BOOST_COPY_ASSIGN_REF(base_type)>(other))
        {}

        function_nonser(BOOST_RV_REF(function_nonser) other)
            : base_type(boost::move(static_cast<BOOST_RV_REF(base_type)>(other)))
        {}

        function_nonser& operator=(BOOST_COPY_ASSIGN_REF(function_nonser) t)
        {
            this->base_type::operator=(t);
            return *this;
        }

        function_nonser& operator=(BOOST_RV_REF(function_nonser) t)
        {
            this->base_type::operator=(boost::move(static_cast<BOOST_RV_REF(base_type)>(t)));
            return *this;
        }

        void clear() { reset(); }

    private:
        BOOST_COPYABLE_AND_MOVABLE(function_nonser)
    };
}}

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/util/detail/preprocessed/function_template.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/function_template_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            0                                                                   \
          , HPX_FUNCTION_ARGUMENT_LIMIT                                         \
          , <hpx/util/detail/function_template.hpp>                             \
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

namespace hpx { namespace util {

    template <
        typename R
      BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, typename A)
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(BOOST_PP_ENUM_PARAMS(N, A))
      , IArchive
      , OArchive
    >
    {
        function_base()
            : vptr(0)
            , object(0)
        {}

        ~function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }

        typedef R result_type;

        typedef
            detail::vtable_ptr_base<
                R(BOOST_PP_ENUM_PARAMS(N, A))
              , IArchive
              , OArchive
            > vtable_ptr_type;

        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;

                vptr = detail::get_table<
                            functor_type
                          , R(BOOST_PP_ENUM_PARAMS(N, A))
                        >::template get<
                            IArchive
                          , OArchive
                        >();

                static const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
        }

        function_base(function_base const & other)
            : vptr(0)
            , object(0)
        {
            assign(other);
        }

        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
            other.object = 0;
        }

        function_base &assign(function_base const & other)
        {
            if(&other != this)
            {
                if(vptr == other.vptr && !empty())
                {
                    vptr->copy(&other.object, &object);
                }
                else
                {
                    reset();
                    if(!other.empty())
                    {
                        other.vptr->clone(&other.object, &object);
                        vptr = other.vptr;
                    }
                }
            }
            return *this;
        }

        template <typename Functor>
        function_base & assign(BOOST_FWD_REF(Functor) f)
        {
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            static const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(BOOST_PP_ENUM_PARAMS(N, A))>::template get<
                    IArchive
                  , OArchive
                >();

            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }

                if (!detail::is_empty_function(f))
                {
                    if(is_small)
                    {
                        new (&object) functor_type(boost::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(boost::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }

        template <typename T>
        function_base & operator=(BOOST_FWD_REF(T) t)
        {
            return assign(boost::forward<T>(t));
        }

        function_base & operator=(BOOST_COPY_ASSIGN_REF(function_base) t)
        {
            return assign(t);
        }

        function_base & operator=(BOOST_RV_REF(function_base) t)
        {
            if(this != &t)
            {
                reset();
                vptr = t.vptr;
                object = t.object;
                t.vptr = 0;
                t.object = 0;
            }

            return *this;
        }

        function_base &swap(function_base& f)
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }

        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }

        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }

        bool operator!() const
        {
            return empty();
        }

        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }

        BOOST_FORCEINLINE R operator()(BOOST_PP_ENUM_BINARY_PARAMS(N, A, a)) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a));
        }

        BOOST_FORCEINLINE R operator()(BOOST_PP_ENUM_BINARY_PARAMS(N, A, a))
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a));
        }

    private:
        BOOST_COPYABLE_AND_MOVABLE(function_base);

    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}

#undef N
#endif

