//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_SUPPORTS_RESULT_OF_APR_15_2012_0601PM)
#define HPX_TRAITS_SUPPORTS_RESULT_OF_APR_15_2012_0601PM

#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/is_function.hpp>
#include <boost/type_traits/remove_pointer.hpp>

#include <boost/mpl/and.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/has_xxx.hpp>

namespace hpx { namespace traits
{
    namespace detail
    {
        template<typename T>
        struct is_function_pointer
          : boost::mpl::and_<
                boost::is_pointer<T>
              , boost::is_function<typename boost::remove_pointer<T>::type>
            >
        {
            typedef is_function_pointer<T> type;
        };

        BOOST_MPL_HAS_XXX_TRAIT_DEF(result_type)
        BOOST_MPL_HAS_XXX_TRAIT_DEF(result)

        template<typename T>
        struct supports_result_of
          : boost::mpl::or_<
                has_result_type<T>
              , has_result<T>
            >
        {
            typedef supports_result_of<T> type;
        };
    }

    template<typename T>
    struct supports_result_of
      : boost::mpl::or_<
            detail::supports_result_of<T>
          , detail::is_function_pointer<T>
        >
    {
        typedef supports_result_of<T> type;
    };
}}

#endif

