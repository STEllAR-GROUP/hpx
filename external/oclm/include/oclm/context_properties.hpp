
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_CONTEXT_PROPERTIES_HPP
#define OCLM_CONTEXT_PROPERTIES_HPP

#include <map>

#include <boost/assert.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_params_with_a_default.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/set.hpp>
#include <boost/mpl/has_key.hpp>

#include <oclm/device.hpp>

#include <CL/cl.h>

#define OCLM_CONTEXT_PROPERTIES_LIMIT 12

namespace oclm
{
    template <typename Properties>
    struct is_context_property
        : boost::mpl::false_
    {};

    namespace detail
    {
        template <int Property, typename T>
        struct context_property
        {
            context_property(T const & t) : value(t), type(Property) {}
            
            T value;
            int type;
        };

        template <
            BOOST_PP_ENUM_PARAMS_WITH_A_DEFAULT(
                OCLM_CONTEXT_PROPERTIES_LIMIT
              , typename T
              , void
            )
          , typename Dummy = void
        >
        struct unique_properties;

        template <typename T>
        struct unique_properties<T>
            : boost::mpl::true_
        {};

        template <typename T1, typename T2>
        struct unique_properties<T1, T2>
            : boost::mpl::true_
        {};

        template <typename T>
        struct unique_properties<T, T>
            : boost::mpl::false_
        {};

#define OCLM_UNIQUE_PROPERTIES(Z, N, D)                                         \
        template <BOOST_PP_ENUM_PARAMS(N, typename T)>                          \
        struct unique_properties<BOOST_PP_ENUM_PARAMS(N, T)>                    \
            : boost::mpl::and_<                                                 \
                unique_properties<BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T)>     \
              , typename boost::mpl::not_<                                      \
                    typename boost::mpl::has_key<                               \
                        boost::mpl::set<BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), T)>\
                      , BOOST_PP_CAT(T, BOOST_PP_DEC(N))                        \
                    >::type                                                     \
                >::type                                                         \
            >::type                                                             \
        {                                                                       \
        };                                                                      \
        /**/
        BOOST_PP_REPEAT_FROM_TO(3, OCLM_CONTEXT_PROPERTIES_LIMIT, OCLM_UNIQUE_PROPERTIES, _)
#undef OCLM_UNIQUE_PROPERTIES

        template <typename T>
        void set_context_property(T const & t, std::map<int, cl_context_properties> & p)
        {
            std::map<int, cl_context_properties>::iterator pos = p.find(t.type);

            /*
            BOOST_ASSERT(pos != p.end());
            
            p.insert(pos, std::make_pair(t.type, reinterpret_cast<cl_context_properties>(t.value)));
            */
        }
    }

    template <int Property, typename T>
    struct is_context_property<detail::context_property<Property, T> >
        : boost::mpl::true_
    {};
    
    #define OCLM_DEFINE_CONTEXT_PROPERTY(PROP, T, NAME)                         \
        typedef detail::context_property<PROP, T> NAME                          \
    /**/
    #define OCLM_DEFINE_CONTEXT_PROPERTIES_I(R, D, E)                           \
        OCLM_DEFINE_CONTEXT_PROPERTY(                                           \
            BOOST_PP_TUPLE_ELEM(3, 0, E)                                        \
          , BOOST_PP_TUPLE_ELEM(3, 1, E)                                        \
          , BOOST_PP_TUPLE_ELEM(3, 2, E)                                        \
        );                                                                      \
    /**/
    #define OCLM_DEFINE_CONTEXT_PROPERTIES(PROP_SEQ)                            \
        BOOST_PP_SEQ_FOR_EACH(OCLM_DEFINE_CONTEXT_PROPERTIES_I, _, PROP_SEQ)    \
    /**/
    OCLM_DEFINE_CONTEXT_PROPERTIES(
        ((CL_CONTEXT_PLATFORM          , cl_platform_id, context_platform))
    )
#ifdef CL_VERSION_1_2
    OCLM_DEFINE_CONTEXT_PROPERTIES(
        ((CL_CONTEXT_INTEROP_USER_SYNC , cl_bool       , context_interop_user_sync))
    )
#endif
    /* FIXME: add proper defines and types.
        ((CL_CONTEXT_D3D10_DEVICE_KHR  , ???           , context_d3d10_device_khr))
        ((CL_CONTEXT_D3D11_DEVICE_KHR  , ???           , context_d3d11_device_khr))
        ((CL_CONTEXT_ADAPTER_D3D9_KHR  , ???           , context_d3d9_adapter_khr))
        ((CL_CONTEXT_ADAPTER_D3D9EX_KHR, ???           , context_d3d9ex_adapter_khr))
        ((CL_CONTEXT_ADAPTER_DXVA_KHR  , ???           , context_dxva_adapter_khr))
        ((CL_CL_CONTEXT_KHR            , ???           , gl_context_khr))
        ((CL_CGL_SHAREGROUP_KHR        , ???           , cgl_sharegroup_khr))
        ((CL_EGL_DISPLAY_KHR           , ???           , egl_display_khr))
        ((CL_GLX_DISPLAY_KHR           , ???           , glx_display_khr))
        ((CL_WGL_HDC_KHR               , ???           , wgl_hdc_khr))
    */
    #undef OCLM_DEFINE_CONTEXT_PROPERTY
    #undef OCLM_DEFINE_CONTEXT_PROPERTIES_I
    #undef OCLM_DEFINE_CONTEXT_PROPERTIES

    inline std::map<int, cl_context_properties> context_properties()
    {
        return std::map<int, cl_context_properties>();
    }
#define OCLM_CHECK_CONTEXT_PROPERTY(Z, N, D)                                    \
        static_assert(                                                          \
            is_context_property<BOOST_PP_CAT(T, N)>::value                      \
          , "context_properties"                                                \
            BOOST_PP_STRINGIZE((BOOST_PP_ENUM_BINARY_PARAMS(N, T, const & t)))  \
            ": Argument " BOOST_PP_STRINGIZE(BOOST_PP_DEC(N))                   \
            " is not a context property"                                        \
        );                                                                      \
/**/

#define OCLM_CONTEXT_PROPERTIES(Z, N, D)                                        \
    template <BOOST_PP_ENUM_PARAMS(N, typename T)>                              \
    inline std::map<int, cl_context_properties>                                 \
    context_properties(BOOST_PP_ENUM_BINARY_PARAMS(N, T, const & t))            \
    {                                                                           \
        static_assert(                                                          \
            detail::unique_properties<BOOST_PP_ENUM_PARAMS(N, T)>::value        \
          , "context_properties: multiple context properties not allowed");     \
        std::map<int, cl_context_properties>                                    \
            ret(context_properties(BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), t)));  \
        BOOST_PP_REPEAT(N, OCLM_CHECK_CONTEXT_PROPERTY, _)                      \
        detail::set_context_property(BOOST_PP_CAT(t, BOOST_PP_DEC(N)), ret);    \
        return ret;                                                             \
    }                                                                           \
/**/
    BOOST_PP_REPEAT_FROM_TO(1, OCLM_CONTEXT_PROPERTIES_LIMIT, OCLM_CONTEXT_PROPERTIES, _)
#undef OCLM_CHECK_CONTEXT_PROPERTIY
#undef OCLM_CONTEXT_PROPERTIES
}

#endif
