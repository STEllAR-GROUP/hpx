/*=============================================================================
    Copyright (c) 2007 Tobias Schwinger
  
    Use modification and distribution are subject to the Boost Software 
    License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
==============================================================================*/

// Purpose:
// 
//      Function object template to return from operator->*. Binds 'this', 
//      providing optimal forwarding.
//
// Example:
//
//      template< typename MP >
//      inline typename detail::member_dereference<Pointee,MP>::type 
//      operator->*(MP mp) const
//      {
//          return detail::member_dereference<Pointee,MP>(ptr_that,mp);
//      }

#ifndef BOOST_UTILITY_DETAIL_MEMBER_DEREFERENCE_HPP_INCLUDED
#   ifndef BOOST_PP_IS_ITERATING

#     include <boost/preprocessor/cat.hpp>
#     include <boost/preprocessor/iteration/iterate.hpp>
#     include <boost/preprocessor/repetition/enum_shifted.hpp>
#     include <boost/preprocessor/repetition/enum_shifted_params.hpp>

#     include <boost/function_types/is_member_function_pointer.hpp>
#     include <boost/function_types/function_arity.hpp>
#     include <boost/function_types/result_type.hpp>
#     include <boost/function_types/parameter_types.hpp>
#     include <boost/call_traits.hpp>
#     include <boost/mpl/at.hpp>

#     ifndef BOOST_UTILITY_MEMBER_DEREFERENCE_MAX_ARITY
#       define BOOST_UTILITY_MEMBER_DEREFERENCE_MAX_ARITY 10
#     elif BOOST_UTILITY_MEMBER_DEREFERENCE_MAX_ARITY < 3
#       undef BOOST_UTILITY_MEMBER_DEREFERENCE_MAX_ARITY
#       define BOOST_UTILITY_MEMBER_DEREFERENCE_MAX_ARITY 3
#     endif

namespace boost { namespace detail
{
    template< class C, typename MP, 
        std::size_t Arity = ::boost::function_types::function_arity<MP>::value,
        bool IsMFP =
            ::boost::function_types::is_member_function_pointer<MP>::value >
    class member_dereference;

    template< class C, typename T, class MPC>
    class member_dereference<C, T MPC::*, 1, false> 
    {
        C* ptr_that;
        T MPC::* ptr_member;
      public:
        member_dereference(C* that, T MPC::* member)
          : ptr_that(that), ptr_member(member)
        { }

        // T C::* is *not* a typo here, but to calculate proper constness
        typedef typename function_types::result_type<T C::*>::type type; 

        inline operator type() const
        {
            return ptr_that->*ptr_member;
        }
    };

#     define BOOST_PP_FILENAME_1 <boost/utility/detail/member_dereference.hpp>
#     define BOOST_PP_ITERATION_LIMITS \
        (1,BOOST_UTILITY_MEMBER_DEREFERENCE_MAX_ARITY+1)
#     include BOOST_PP_ITERATE()

}}

#     define BOOST_UTILITY_DETAIL_MEMBER_DEREFERENCE_HPP_INCLUDED
#   else // defined(BOOST_PP_IS_ITERATING)
#     define N BOOST_PP_ITERATION()

    template< class C, typename MP>
    class member_dereference<C, MP, N, true>
    {
        C* ptr_that;
        MP ptr_member;

        typedef function_types::parameter_types<MP> parameter_types;
      public:
        member_dereference(C* that, MP member)
          : ptr_that(that), ptr_member(member)
        { }

        typedef typename function_types::result_type<MP>::type result_type;

#     define M(z,i,d) typename boost::call_traits< typename mpl::at_c< \
            parameter_types, i >::type >::param_type BOOST_PP_CAT(a,i)

        inline result_type operator()( BOOST_PP_ENUM_SHIFTED(N,M,~) ) const
        {
            return (ptr_that->*ptr_member)(BOOST_PP_ENUM_SHIFTED_PARAMS(N,a));
        }

#     undef M

        typedef member_dereference type;
    };

#     undef N
#   endif // defined(BOOST_PP_IS_ITERATING)
#endif // include guard


