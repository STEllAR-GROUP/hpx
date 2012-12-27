//  boost win32_test.cpp  -----------------------------------------------------//

//  Copyright 2010 Vicente J. Botet Escriba

//  Distributed under the Boost Software License, Version 1.0.
//  See http://www.boost.org/LICENSE_1_0.txt

//  See http://www.boost.org/libs/chrono for documentation.
#include <boost/chrono/config.hpp>
#include <boost/detail/lightweight_test.hpp>
#if defined(BOOST_CHRONO_WINDOWS_API) ||  defined(__CYGWIN__)

#include <boost/chrono/detail/static_assert.hpp>
#if !defined(BOOST_NO_CXX11_STATIC_ASSERT)
#define NOTHING ""
#endif

#include <boost/type_traits.hpp>
#include <boost/typeof/typeof.hpp>
#undef BOOST_USE_WINDOWS_H
#include <boost/detail/win/basic_types.hpp>
#include <boost/detail/win/time.hpp>
#include <windows.h>

void test() {
    {
    boost::detail::win32::LARGE_INTEGER_ a;
    LARGE_INTEGER b;
    BOOST_CHRONO_STATIC_ASSERT((
            sizeof(boost::detail::win32::LARGE_INTEGER_)==sizeof(LARGE_INTEGER)
        ), NOTHING, (boost::detail::win32::LARGE_INTEGER_, LARGE_INTEGER));
    BOOST_TEST((
            sizeof(a.QuadPart)==sizeof(b.QuadPart)
            ));
    BOOST_CHRONO_STATIC_ASSERT((
            offsetof(boost::detail::win32::LARGE_INTEGER_, QuadPart)==offsetof(LARGE_INTEGER, QuadPart)
        ), NOTHING, (boost::detail::win32::LARGE_INTEGER_, LARGE_INTEGER));
    BOOST_CHRONO_STATIC_ASSERT((
            boost::is_same<
                    BOOST_TYPEOF(a.QuadPart),
                    BOOST_TYPEOF(b.QuadPart)
                >::value
        ), NOTHING, (boost::detail::win32::LARGE_INTEGER_, LARGE_INTEGER));
    }

    BOOST_CHRONO_STATIC_ASSERT((
            sizeof(boost::detail::win32::BOOL_)==sizeof(BOOL)
        ), NOTHING, (boost::detail::win32::BOOL_, BOOL));
    BOOST_CHRONO_STATIC_ASSERT((
            boost::is_same<boost::detail::win32::BOOL_,BOOL>::value
        ), NOTHING, (boost::detail::win32::BOOL_, BOOL));

    BOOST_CHRONO_STATIC_ASSERT((
            sizeof(boost::detail::win32::DWORD_)==sizeof(DWORD)
        ), NOTHING, (boost::detail::win32::DWORD_, DWORD));
    BOOST_CHRONO_STATIC_ASSERT((
            boost::is_same<boost::detail::win32::DWORD_,DWORD>::value
        ), NOTHING, (boost::detail::win32::DWORD_, DWORD));

    BOOST_CHRONO_STATIC_ASSERT((
            sizeof(boost::detail::win32::HANDLE_)==sizeof(HANDLE)
        ), NOTHING, (boost::detail::win32::HANDLE_, HANDLE));
    BOOST_CHRONO_STATIC_ASSERT((
            boost::is_same<boost::detail::win32::HANDLE_,HANDLE>::value
        ), NOTHING, (boost::detail::win32::HANDLE_, HANDLE));

    BOOST_CHRONO_STATIC_ASSERT((
            sizeof(boost::detail::win32::LONG_)==sizeof(LONG)
        ), NOTHING, (boost::detail::win32::LONG_, LONG));
    BOOST_CHRONO_STATIC_ASSERT((
            boost::is_same<boost::detail::win32::LONG_,LONG>::value
        ), NOTHING, (boost::detail::win32::LONG_, LONG));

    BOOST_CHRONO_STATIC_ASSERT((
            sizeof(boost::detail::win32::LONGLONG_)==sizeof(LONGLONG)
        ), NOTHING, (boost::detail::win32::LONGLONG_, LONGLONG));
    BOOST_CHRONO_STATIC_ASSERT((
            boost::is_same<boost::detail::win32::LONGLONG_,LONGLONG>::value
        ), NOTHING, (boost::detail::win32::LONGLONG_, LONGLONG));

    BOOST_CHRONO_STATIC_ASSERT((
            sizeof(boost::detail::win32::ULONG_PTR_)==sizeof(ULONG_PTR)
        ), NOTHING, (boost::detail::win32::ULONG_PTR_, ULONG_PTR));
    BOOST_CHRONO_STATIC_ASSERT((
            boost::is_same<boost::detail::win32::ULONG_PTR_,ULONG_PTR>::value
        ), NOTHING, (boost::detail::win32::ULONG_PTR_, ULONG_PTR));
        
    BOOST_CHRONO_STATIC_ASSERT((
            sizeof(boost::detail::win32::PLARGE_INTEGER_)==sizeof(PLARGE_INTEGER)
        ), NOTHING, (boost::detail::win32::PLARGE_INTEGER_, PLARGE_INTEGER));
    //~ BOOST_CHRONO_STATIC_ASSERT((
            //~ boost::is_same<boost::detail::win32::PLARGE_INTEGER_,PLARGE_INTEGER>::value
        //~ ), NOTHING, (boost::detail::win32::PLARGE_INTEGER_, PLARGE_INTEGER));
        
    {
        BOOST_CHRONO_STATIC_ASSERT((
                sizeof(boost::detail::win32::FILETIME_)==sizeof(FILETIME)
            ), NOTHING, (boost::detail::win32::FILETIME_, FILETIME));
        
        BOOST_CHRONO_STATIC_ASSERT((
                sizeof(boost::detail::win32::PFILETIME_)==sizeof(PFILETIME)
            ), NOTHING, (boost::detail::win32::PFILETIME_, PFILETIME));
        

        boost::detail::win32::FILETIME_ a;
        FILETIME b;
        BOOST_TEST((
                sizeof(a.dwLowDateTime)==sizeof(b.dwLowDateTime)
                ));
        BOOST_TEST((
                sizeof(a.dwHighDateTime)==sizeof(b.dwHighDateTime)
                ));
        BOOST_CHRONO_STATIC_ASSERT((
                offsetof(boost::detail::win32::FILETIME_, dwLowDateTime)==offsetof(FILETIME, dwLowDateTime)
            ), NOTHING, (boost::detail::win32::FILETIME_, FILETIME));
        BOOST_CHRONO_STATIC_ASSERT((
                offsetof(boost::detail::win32::FILETIME_, dwHighDateTime)==offsetof(FILETIME, dwHighDateTime)
            ), NOTHING, (boost::detail::win32::FILETIME_, FILETIME));
        BOOST_CHRONO_STATIC_ASSERT((
            boost::is_same<
                    BOOST_TYPEOF(a.dwLowDateTime),
                    BOOST_TYPEOF(b.dwLowDateTime)
                >::value
        ), NOTHING, (boost::detail::win32::FILETIME_, FILETIME));
        BOOST_CHRONO_STATIC_ASSERT((
            boost::is_same<
                    BOOST_TYPEOF(a.dwHighDateTime),
                    BOOST_TYPEOF(b.dwHighDateTime)
                >::value
        ), NOTHING, (boost::detail::win32::FILETIME_, FILETIME));

    }

//    BOOST_CHRONO_STATIC_ASSERT((
//            GetLastError==boost::detail::win32::::GetLastError
//        ), NOTHING, ());

}
#else
void test() {
}
#endif
int main(  )
{
    test();

  return boost::report_errors();
}

