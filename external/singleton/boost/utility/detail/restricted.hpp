/*=============================================================================
    Copyright (c) 2007 Tobias Schwinger
  
    Use modification and distribution are subject to the Boost Software 
    License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
==============================================================================*/

// Purpose:
// 
//      Restrict access to otherwise public member functions or constructors, 
//      using a special parameter 'restricted', to specific, boost-side clients.
//
// Example:
//
//      struct X
//      {
//          void framework_only(restricted);
//      };
//
//      template<class T> 
//      struct Y
//      {
//          // Note: Can't make friends with T and inheritance might be unsuitable
//          // to gain access
//
//          void f()
//          {
//              y.framework_only(detail::restricted_argument());
//          }
//      };

#ifndef BOOST_UTILITY_DETAIL_RESTRICTED_HPP_INCLUDED
#   define BOOST_UTILITY_DETAIL_RESTRICTED_HPP_INCLUDED

namespace boost 
{
    namespace detail
    { 
        class restrictor : boost::noncopyable
        {
            restrictor()
            { }

            friend inline restrictor const & restricted_argument();
        };

        inline restrictor const & restricted_argument()
        {
            static restrictor const result;
            return result;
        } 
    }

    typedef detail::restrictor const & restricted;
}

#endif

