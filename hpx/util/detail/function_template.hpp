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
#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <hpx/util/detail/vtable_ptr_base_fwd.hpp>
#include <hpx/util/detail/vtable_ptr_fwd.hpp>
#include <hpx/util/detail/serialization_registration.hpp>

#ifndef HPX_FUNCTION_VERSION
#define HPX_FUNCTION_VERSION 0x10
#endif

#define HPX_FUNCTION_NAME function

namespace hpx { namespace util {
    namespace detail
    {
        template <
            typename Functor
          , typename Sig
        >
        struct get_table;

        template <bool>
        struct vtable;
    }

    template <
        typename Sig
#if HPX_USE_PORTABLE_ARCHIVES != 0
      , typename IArchive = portable_binary_iarchive
      , typename OArchive = portable_binary_oarchive
#else
      , typename IArchive = boost::archive::binary_iarchive
      , typename OArchive = boost::archive::binary_oarchive
#endif
    >
    struct HPX_FUNCTION_NAME;

}}

namespace boost { namespace serialization {
    template <typename Sig, typename IArchive, typename OArchive>
    struct tracking_level<hpx::util::HPX_FUNCTION_NAME<Sig, IArchive, OArchive> >
    {
        typedef mpl::integral_c_tag tag;
        typedef mpl::int_<track_never> type;
        BOOST_STATIC_CONSTANT(int, value = track_never);
    };
    
    template <typename Sig, typename IArchive, typename OArchive>
    struct version<hpx::util::HPX_FUNCTION_NAME<Sig, IArchive, OArchive> >
    {
        typedef mpl::integral_c_tag tag;
        typedef mpl::int_<HPX_FUNCTION_VERSION> type;
        BOOST_STATIC_CONSTANT(int, value = HPX_FUNCTION_VERSION);
    };
}}

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            0                                                                   \
          , HPX_FUNCTION_LIMIT                                                  \
          , <hpx/util/detail/function_template.hpp>                             \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

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
    struct HPX_FUNCTION_NAME<
        R(BOOST_PP_ENUM_PARAMS(N, A))
      , IArchive
      , OArchive
    >
    {
        HPX_FUNCTION_NAME()
            : vptr(0)
            , object(0)
        {}
        
        ~HPX_FUNCTION_NAME()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        
        typedef
            detail::vtable_ptr_base<
                R(BOOST_PP_ENUM_PARAMS(N, A))
              , IArchive
              , OArchive
            > vtable_ptr_type;
        
        template <typename Functor>
            HPX_FUNCTION_NAME(Functor f)
            : vptr(
                detail::get_table<Functor, R(BOOST_PP_ENUM_PARAMS(N, A))>::template get<
                    IArchive
                  , OArchive
                >()
            )
            , object(0)
        {
            static const bool is_small = sizeof(Functor) <= sizeof(void *);
            if(is_small)
            {
                new (&object) Functor(f);
            }
            else
            {
                object = new Functor(f);
            }
        }
        
        HPX_FUNCTION_NAME(HPX_FUNCTION_NAME const& other)
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        
        HPX_FUNCTION_NAME &assign(HPX_FUNCTION_NAME const & other)
        {
            if(&other != this)
            {
                if(vptr == other.vptr && !empty())
                {
                    vptr->move(&other.object, &object);
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
        HPX_FUNCTION_NAME & assign(Functor const & f)
        {
            static const bool is_small = sizeof(Functor) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<Functor, R(BOOST_PP_ENUM_PARAMS(N, A))>::template get<
                    IArchive
                  , OArchive
                >();
            
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) Functor(f);
                }
                else
                {
                    new (object) Functor(f);
                }
            }
            else
            {
                if(is_small)
                {
                    if(!empty())
                    {
                        vptr->destruct(&object);
                    }
                    new (&object) Functor(f);
                }
                else
                {
                    reset();
                    object = new Functor(f);
                }
                vptr = f_vptr;
            }
            return *this;
        }
        
        template <typename T>
        HPX_FUNCTION_NAME & operator=(T const & t)
        {
            return assign(t);
        }
        
        HPX_FUNCTION_NAME &swap(HPX_FUNCTION_NAME& f)
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
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
        
        R operator()(BOOST_PP_ENUM_BINARY_PARAMS(N, A, a))
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object BOOST_PP_COMMA_IF(N) BOOST_PP_ENUM_PARAMS(N, a));
        }
        
        void serialize(IArchive &ar, const unsigned version)
        {
            bool is_empty;
            ar & is_empty;

            if(is_empty)
            {
                reset();
            }
            else
            {
                vtable_ptr_type *p = 0;
                ar & p;
                vptr = p->get_ptr();
                delete p;
                vptr->iserialize(&object, ar, version);
            }
        }
        
        void serialize(OArchive &ar, const unsigned version)
        {
            bool is_empty = empty();
            ar & is_empty;
            if(!empty())
            {
                ar & vptr;
                vptr->oserialize(&object, ar, version);
            }
        }
        
        vtable_ptr_type *vptr;
        void *object;
    };
}}

#endif

