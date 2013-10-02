//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !BOOST_PP_IS_ITERATING

#ifndef HPX_FUNCTION_DETAIL_VTABLE_PTR_BASE_HPP
#define HPX_FUNCTION_DETAIL_VTABLE_PTR_BASE_HPP

#include <hpx/config.hpp>
#include <hpx/util/detail/vtable_ptr_base_fwd.hpp>
#include <hpx/util/add_rvalue_reference.hpp>
#include <hpx/util/polymorphic_factory.hpp>
#include <hpx/util/demangle_helper.hpp>
#include <hpx/util/move.hpp>

#include <boost/detail/sp_typeinfo.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/assert.hpp>

namespace hpx { namespace util { namespace detail {

    /////////////////////////////////////////////////////////////////////////////
    template <
        typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_virtbase
    {
        virtual ~vtable_ptr_virtbase() {}

        virtual vtable_ptr_virtbase* get_ptr() = 0;
        virtual char const* get_function_name() const = 0;
        virtual bool empty() const = 0;

        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
    };

    /////////////////////////////////////////////////////////////////////////////
    template <typename Function>
    char const* get_function_name()
#ifdef HPX_DISABLE_AUTOMATIC_SERIALIZATION_REGISTRATION
    ;
#else
    {
        // If you encounter this assert while compiling code, that means that
        // you have a HPX_UTIL_REGISTER_FUNCTION macro somewhere in a
        // source file, but the header in which the continuation is defined
        // misses a HPX_UTIL_REGISTER_FUNCTION_DECLARATION
        BOOST_MPL_ASSERT_MSG(
            traits::needs_automatic_registration<Function>::value
            , HPX_UTIL_REGISTER_FUNCTION_DECLARATION_MISSING
            , (Function)
        );
        return util::type_id<Function>::typeid_.type_id();
    }
#endif

    /////////////////////////////////////////////////////////////////////////////
    template <typename Function>
    struct function_registration
    {
        typedef
            typename Function::vtable_ptr_virtbase_type
            vtable_ptr_virtbase_type;

        static boost::shared_ptr<vtable_ptr_virtbase_type> create()
        {
            return boost::shared_ptr<vtable_ptr_virtbase_type>(new Function());
        }

        function_registration()
        {
            util::polymorphic_factory<vtable_ptr_virtbase_type>::get_instance().
                add_factory_function(
                    detail::get_function_name<Function>()
                  , &function_registration::create
                );
        }
    };

    template <typename Function, typename Enable =
        typename traits::needs_automatic_registration<Function>::type>
    struct automatic_function_registration
    {
        automatic_function_registration()
        {
            function_registration<Function> auto_register;
        }

        automatic_function_registration& register_function()
        {
            return *this;
        }
    };

    template <typename Function>
    struct automatic_function_registration<Function, boost::mpl::false_>
    {
        automatic_function_registration()
        {
        }

        automatic_function_registration& register_function()
        {
            return *this;
        }
    };

}}}

namespace boost { namespace serialization {

    template <
        typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_virtbase<
        IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};

}}

#define BOOST_UTIL_DETAIL_VTABLE_PTR_BASE_ADD_RVALUE_REF(Z, N, D)               \
    typename util::add_rvalue_reference<BOOST_PP_CAT(D, N)>::type               \
    /**/

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
#  include <hpx/util/detail/preprocessed/vtable_ptr_base.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/vtable_ptr_base_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                             \
    (                                                                           \
        3                                                                       \
      , (                                                                       \
            0                                                                   \
          , HPX_FUNCTION_ARGUMENT_LIMIT                                         \
          , <hpx/util/detail/vtable_ptr_base.hpp>                               \
        )                                                                       \
    )                                                                           \
/**/
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

#undef BOOST_UTIL_DETAIL_VTABLE_PTR_BASE_ADD_RVALUE_REF

#endif

#else

#define N BOOST_PP_ITERATION()

namespace hpx { namespace util { namespace detail {

    template <
        typename R
      BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(BOOST_PP_ENUM_PARAMS(N, A))
      , IArchive
      , OArchive
    >
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}

        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            BOOST_PP_ENUM_TRAILING(N, BOOST_UTIL_DETAIL_VTABLE_PTR_BASE_ADD_RVALUE_REF, A));
    };

    ///////////////////////////////////////////////////////////////////////////
    template <
        typename R
      BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
    >
    struct vtable_ptr_base<
        R(BOOST_PP_ENUM_PARAMS(N, A))
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual bool empty() const = 0;

        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            BOOST_PP_ENUM_TRAILING(N, BOOST_UTIL_DETAIL_VTABLE_PTR_BASE_ADD_RVALUE_REF, A));
    };
}}}

namespace boost { namespace serialization {

    template <
        typename R
      BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(BOOST_PP_ENUM_PARAMS(N, A)), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};

}}

#undef N
#endif
