//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_UTIL_FUNCTION_TEMPLATE_HPP
#define HPX_UTIL_FUNCTION_TEMPLATE_HPP

#include <hpx/config/forceinline.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>
#include <hpx/util/detail/vtable_ptr_base_fwd.hpp>
#include <hpx/util/detail/vtable_ptr_fwd.hpp>
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

namespace hpx { namespace util
{
    template <
        typename Sig
      , typename IArchive = portable_binary_iarchive
      , typename OArchive = portable_binary_oarchive
    >
    struct function;

    template <
        typename Sig
    >
    struct function_nonser;

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Sig
          , typename IArchive = void, typename OArchive = void
        >
        struct function_base;

        ///////////////////////////////////////////////////////////////////////
        template <
            typename Functor
          , typename Sig
        >
        struct get_table;

        template <bool>
        struct unique_vtable;

        template <bool>
        struct vtable;

        ///////////////////////////////////////////////////////////////////////
        template <typename Sig>
        struct get_empty_table;

        ///////////////////////////////////////////////////////////////////////
        template <typename Functor>
        struct is_empty_function_impl
        {
            // in the general case the functor is not empty
            static bool call(Functor const&, boost::mpl::false_) BOOST_NOEXCEPT
            {
                return false;
            }

            static bool call(Functor const& f, boost::mpl::true_) BOOST_NOEXCEPT
            {
                return f == 0;
            }

            static bool call(Functor const& f) BOOST_NOEXCEPT
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
        bool is_empty_function(Functor const& f) BOOST_NOEXCEPT
        {
            return is_empty_function_impl<Functor>::call(f);
        }

        template <
            typename Sig
            , typename IArchive
            , typename OArchive
        >
        bool is_empty_function(function<Sig, IArchive, OArchive> const& f) BOOST_NOEXCEPT
        {
            return f.empty();
        }

        template <
            typename Sig
        >
        bool is_empty_function(function_nonser<Sig> const& f) BOOST_NOEXCEPT
        {
            return f.empty();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig, typename IArchive, typename OArchive>
    struct function : detail::function_base<Sig, IArchive, OArchive>
    {
        typedef detail::function_base<Sig, IArchive, OArchive> base_type;
        typedef typename base_type::result_type result_type;

        function() BOOST_NOEXCEPT
          : base_type() {}

        template <typename Functor>
        function(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function
                  , typename util::decay<Functor>::type
                >::type
            >::type * = 0
        ) : base_type(std::forward<Functor>(f))
        {}

        function(function const & other)
          : base_type(static_cast<base_type const &>(other))
        {}

        function(function && other) BOOST_NOEXCEPT
          : base_type(std::move(static_cast<base_type &&>(other)))
        {}

        function& operator=(function const & t)
        {
            this->base_type::operator=(static_cast<base_type const &>(t));
            return *this;
        }

        function& operator=(function && t) BOOST_NOEXCEPT
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
    struct function<Sig, void, void> : detail::function_base<Sig, void, void>
    {
        typedef detail::function_base<Sig, void, void> base_type;
        typedef typename base_type::result_type result_type;

        function() BOOST_NOEXCEPT
          : base_type() {}

        template <typename Functor>
        function(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function
                  , typename util::decay<Functor>::type
                >::type
            >::type * = 0
        ) : base_type(std::forward<Functor>(f))
        {}

        function(function const & other)
          : base_type(static_cast<base_type const &>(other))
        {}

        function(function && other) BOOST_NOEXCEPT
          : base_type(std::move(static_cast<base_type &&>(other)))
        {}

        function& operator=(function const & t)
        {
            this->base_type::operator=(static_cast<base_type const &>(t));
            return *this;
        }

        function& operator=(function && t) BOOST_NOEXCEPT
        {
            this->base_type::operator=(std::move(static_cast<base_type &&>(t)));
            return *this;
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig>
    struct function_nonser : detail::function_base<Sig, void, void>
    {
        typedef detail::function_base<Sig, void, void> base_type;
        typedef typename base_type::result_type result_type;

        function_nonser() BOOST_NOEXCEPT
          : base_type() {}

        template <typename Functor>
        function_nonser(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_nonser
                  , typename util::decay<Functor>::type
                >::type
            >::type * = 0
        ) : base_type(std::forward<Functor>(f))
        {}

        function_nonser(function_nonser const & other)
          : base_type(static_cast<base_type const &>(other))
        {}

        function_nonser(function_nonser && other) BOOST_NOEXCEPT
          : base_type(std::move(static_cast<base_type &&>(other)))
        {}

        function_nonser& operator=(function_nonser const & t)
        {
            this->base_type::operator=(static_cast<base_type const &>(t));
            return *this;
        }

        function_nonser& operator=(function_nonser && t) BOOST_NOEXCEPT
        {
            this->base_type::operator=(std::move(static_cast<base_type &&>(t)));
            return *this;
        }
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

namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
      , typename IArchive, typename OArchive
    >
    struct function_base<
        R(BOOST_PP_ENUM_PARAMS(N, A))
      , IArchive, OArchive
    >
    {
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
        
        function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}

        ~function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }

        template <typename Functor>
        explicit function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
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

        function_base(function_base const & other)
          : vptr(get_empty_table_ptr())
          , object(0)
        {
            assign(other);
        }

        function_base(function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
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
        function_base & assign(Functor && f)
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
        function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }

        function_base & operator=(function_base const & t)
        {
            return assign(t);
        }

        function_base & operator=(function_base && t) BOOST_NOEXCEPT
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

        function_base &swap(function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }

        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }

        operator typename util::safe_bool<function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<function_base>()(!empty());
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
                    >::template get<false, IArchive, OArchive>();
        }

        BOOST_FORCEINLINE R operator()(BOOST_PP_ENUM_BINARY_PARAMS(N, A, a)) const
        {
            return vptr->invoke(&object
                BOOST_PP_COMMA_IF(N) HPX_ENUM_FORWARD_ARGS(N, A, a));
        }

        std::type_info const& target_type() const BOOST_NOEXCEPT
        {
            return vptr->get_type();
        }

        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;

            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;

            if (sizeof(functor_type) <= sizeof(void *))  // is_small
                return reinterpret_cast<functor_type *>(&object);

            return reinterpret_cast<functor_type *>(object);
        }

        template <typename T>
        T const* target() const BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;

            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;

            if (sizeof(functor_type) <= sizeof(void *))  // is_small
                return reinterpret_cast<functor_type const*>(&object);

            return reinterpret_cast<functor_type const*>(object);
        }

    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}}

#undef N
#endif
