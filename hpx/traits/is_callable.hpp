//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_CALLABLE_APR_15_2012_0601PM)
#define HPX_TRAITS_IS_CALLABLE_APR_15_2012_0601PM

#include <boost/type_traits/is_class.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/is_function.hpp>
#include <boost/type_traits/is_member_function_pointer.hpp>
#include <boost/type_traits/remove_pointer.hpp>

#include <boost/mpl/and.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/has_xxx.hpp>

//#include <hpx/traits/has_call_operator.hpp>
#include <hpx/util/detail/remove_reference.hpp>

namespace hpx { namespace traits
{
    namespace detail
    {
        template <typename T>
        struct is_function_pointer
          : boost::mpl::and_<
                boost::is_pointer<T>
              , boost::is_function<typename boost::remove_pointer<T>::type>
            >
        {
            typedef is_function_pointer<T> type;
        };

        BOOST_MPL_HAS_XXX_TRAIT_DEF(result_type)

//         template <typename T, typename Signature>
//         struct is_function_object
//           : boost::mpl::and_<
//                 boost::is_class<T>
//               , boost::mpl::bool_<has_call_operator<T, Signature>::value>
//             >
//         {
//             typedef is_function_object<T, Signature> type;
//         };
    }

    template <typename T/*, typename Signature*/>
    struct is_callable
      : boost::mpl::or_<
            detail::has_result_type<T>
          , detail::is_function_pointer<T>
          , boost::is_member_function_pointer<T>
          , boost::is_function<typename util::detail::remove_reference<T>::type>
//           , detail::is_function_object<
//                 typename util::detail::remove_reference<T>::type, Signature
//             >
        >
    {
        typedef is_callable<T/*, Signature*/> type;
    };
}}

#endif

