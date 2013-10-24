// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R()
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)();
        static vtable_ptr_base<
            R()
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R()
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            )
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0);
        static vtable_ptr_base<
            R(A0)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1);
        static vtable_ptr_base<
            R(A0 , A1)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2);
        static vtable_ptr_base<
            R(A0 , A1 , A2)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3 , A4)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3 , A4);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3 , A4)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3 , typename util::add_rvalue_reference<A4>::type a4)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3 , A4 , A5)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3 , A4 , A5);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3 , A4 , A5)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3 , typename util::add_rvalue_reference<A4>::type a4 , typename util::add_rvalue_reference<A5>::type a5)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3 , A4 , A5 , A6);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3 , typename util::add_rvalue_reference<A4>::type a4 , typename util::add_rvalue_reference<A5>::type a5 , typename util::add_rvalue_reference<A6>::type a6)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3 , typename util::add_rvalue_reference<A4>::type a4 , typename util::add_rvalue_reference<A5>::type a5 , typename util::add_rvalue_reference<A6>::type a6 , typename util::add_rvalue_reference<A7>::type a7)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3 , typename util::add_rvalue_reference<A4>::type a4 , typename util::add_rvalue_reference<A5>::type a5 , typename util::add_rvalue_reference<A6>::type a6 , typename util::add_rvalue_reference<A7>::type a7 , typename util::add_rvalue_reference<A8>::type a8)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3 , typename util::add_rvalue_reference<A4>::type a4 , typename util::add_rvalue_reference<A5>::type a5 , typename util::add_rvalue_reference<A6>::type a6 , typename util::add_rvalue_reference<A7>::type a7 , typename util::add_rvalue_reference<A8>::type a8 , typename util::add_rvalue_reference<A9>::type a9)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3 , typename util::add_rvalue_reference<A4>::type a4 , typename util::add_rvalue_reference<A5>::type a5 , typename util::add_rvalue_reference<A6>::type a6 , typename util::add_rvalue_reference<A7>::type a7 , typename util::add_rvalue_reference<A8>::type a8 , typename util::add_rvalue_reference<A9>::type a9 , typename util::add_rvalue_reference<A10>::type a10)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3 , typename util::add_rvalue_reference<A4>::type a4 , typename util::add_rvalue_reference<A5>::type a5 , typename util::add_rvalue_reference<A6>::type a6 , typename util::add_rvalue_reference<A7>::type a7 , typename util::add_rvalue_reference<A8>::type a8 , typename util::add_rvalue_reference<A9>::type a9 , typename util::add_rvalue_reference<A10>::type a10 , typename util::add_rvalue_reference<A11>::type a11)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3 , typename util::add_rvalue_reference<A4>::type a4 , typename util::add_rvalue_reference<A5>::type a5 , typename util::add_rvalue_reference<A6>::type a6 , typename util::add_rvalue_reference<A7>::type a7 , typename util::add_rvalue_reference<A8>::type a8 , typename util::add_rvalue_reference<A9>::type a9 , typename util::add_rvalue_reference<A10>::type a10 , typename util::add_rvalue_reference<A11>::type a11 , typename util::add_rvalue_reference<A12>::type a12)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3 , typename util::add_rvalue_reference<A4>::type a4 , typename util::add_rvalue_reference<A5>::type a5 , typename util::add_rvalue_reference<A6>::type a6 , typename util::add_rvalue_reference<A7>::type a7 , typename util::add_rvalue_reference<A8>::type a8 , typename util::add_rvalue_reference<A9>::type a9 , typename util::add_rvalue_reference<A10>::type a10 , typename util::add_rvalue_reference<A11>::type a11 , typename util::add_rvalue_reference<A12>::type a12 , typename util::add_rvalue_reference<A13>::type a13)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3 , typename util::add_rvalue_reference<A4>::type a4 , typename util::add_rvalue_reference<A5>::type a5 , typename util::add_rvalue_reference<A6>::type a6 , typename util::add_rvalue_reference<A7>::type a7 , typename util::add_rvalue_reference<A8>::type a8 , typename util::add_rvalue_reference<A9>::type a9 , typename util::add_rvalue_reference<A10>::type a10 , typename util::add_rvalue_reference<A11>::type a11 , typename util::add_rvalue_reference<A12>::type a12 , typename util::add_rvalue_reference<A13>::type a13 , typename util::add_rvalue_reference<A14>::type a14)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3 , typename util::add_rvalue_reference<A4>::type a4 , typename util::add_rvalue_reference<A5>::type a5 , typename util::add_rvalue_reference<A6>::type a6 , typename util::add_rvalue_reference<A7>::type a7 , typename util::add_rvalue_reference<A8>::type a8 , typename util::add_rvalue_reference<A9>::type a9 , typename util::add_rvalue_reference<A10>::type a10 , typename util::add_rvalue_reference<A11>::type a11 , typename util::add_rvalue_reference<A12>::type a12 , typename util::add_rvalue_reference<A13>::type a13 , typename util::add_rvalue_reference<A14>::type a14 , typename util::add_rvalue_reference<A15>::type a15)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3 , typename util::add_rvalue_reference<A4>::type a4 , typename util::add_rvalue_reference<A5>::type a5 , typename util::add_rvalue_reference<A6>::type a6 , typename util::add_rvalue_reference<A7>::type a7 , typename util::add_rvalue_reference<A8>::type a8 , typename util::add_rvalue_reference<A9>::type a9 , typename util::add_rvalue_reference<A10>::type a10 , typename util::add_rvalue_reference<A11>::type a11 , typename util::add_rvalue_reference<A12>::type a12 , typename util::add_rvalue_reference<A13>::type a13 , typename util::add_rvalue_reference<A14>::type a14 , typename util::add_rvalue_reference<A15>::type a15 , typename util::add_rvalue_reference<A16>::type a16)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3 , typename util::add_rvalue_reference<A4>::type a4 , typename util::add_rvalue_reference<A5>::type a5 , typename util::add_rvalue_reference<A6>::type a6 , typename util::add_rvalue_reference<A7>::type a7 , typename util::add_rvalue_reference<A8>::type a8 , typename util::add_rvalue_reference<A9>::type a9 , typename util::add_rvalue_reference<A10>::type a10 , typename util::add_rvalue_reference<A11>::type a11 , typename util::add_rvalue_reference<A12>::type a12 , typename util::add_rvalue_reference<A13>::type a13 , typename util::add_rvalue_reference<A14>::type a14 , typename util::add_rvalue_reference<A15>::type a15 , typename util::add_rvalue_reference<A16>::type a16 , typename util::add_rvalue_reference<A17>::type a17)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3 , typename util::add_rvalue_reference<A4>::type a4 , typename util::add_rvalue_reference<A5>::type a5 , typename util::add_rvalue_reference<A6>::type a6 , typename util::add_rvalue_reference<A7>::type a7 , typename util::add_rvalue_reference<A8>::type a8 , typename util::add_rvalue_reference<A9>::type a9 , typename util::add_rvalue_reference<A10>::type a10 , typename util::add_rvalue_reference<A11>::type a11 , typename util::add_rvalue_reference<A12>::type a12 , typename util::add_rvalue_reference<A13>::type a13 , typename util::add_rvalue_reference<A14>::type a14 , typename util::add_rvalue_reference<A15>::type a15 , typename util::add_rvalue_reference<A16>::type a16 , typename util::add_rvalue_reference<A17>::type a17 , typename util::add_rvalue_reference<A18>::type a18)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3 , typename util::add_rvalue_reference<A4>::type a4 , typename util::add_rvalue_reference<A5>::type a5 , typename util::add_rvalue_reference<A6>::type a6 , typename util::add_rvalue_reference<A7>::type a7 , typename util::add_rvalue_reference<A8>::type a8 , typename util::add_rvalue_reference<A9>::type a9 , typename util::add_rvalue_reference<A10>::type a10 , typename util::add_rvalue_reference<A11>::type a11 , typename util::add_rvalue_reference<A12>::type a12 , typename util::add_rvalue_reference<A13>::type a13 , typename util::add_rvalue_reference<A14>::type a14 , typename util::add_rvalue_reference<A15>::type a15 , typename util::add_rvalue_reference<A16>::type a16 , typename util::add_rvalue_reference<A17>::type a17 , typename util::add_rvalue_reference<A18>::type a18 , typename util::add_rvalue_reference<A19>::type a19)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3 , typename util::add_rvalue_reference<A4>::type a4 , typename util::add_rvalue_reference<A5>::type a5 , typename util::add_rvalue_reference<A6>::type a6 , typename util::add_rvalue_reference<A7>::type a7 , typename util::add_rvalue_reference<A8>::type a8 , typename util::add_rvalue_reference<A9>::type a9 , typename util::add_rvalue_reference<A10>::type a10 , typename util::add_rvalue_reference<A11>::type a11 , typename util::add_rvalue_reference<A12>::type a12 , typename util::add_rvalue_reference<A13>::type a13 , typename util::add_rvalue_reference<A14>::type a14 , typename util::add_rvalue_reference<A15>::type a15 , typename util::add_rvalue_reference<A16>::type a16 , typename util::add_rvalue_reference<A17>::type a17 , typename util::add_rvalue_reference<A18>::type a18 , typename util::add_rvalue_reference<A19>::type a19 , typename util::add_rvalue_reference<A20>::type a20)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20 , typename A21
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3 , typename util::add_rvalue_reference<A4>::type a4 , typename util::add_rvalue_reference<A5>::type a5 , typename util::add_rvalue_reference<A6>::type a6 , typename util::add_rvalue_reference<A7>::type a7 , typename util::add_rvalue_reference<A8>::type a8 , typename util::add_rvalue_reference<A9>::type a9 , typename util::add_rvalue_reference<A10>::type a10 , typename util::add_rvalue_reference<A11>::type a11 , typename util::add_rvalue_reference<A12>::type a12 , typename util::add_rvalue_reference<A13>::type a13 , typename util::add_rvalue_reference<A14>::type a14 , typename util::add_rvalue_reference<A15>::type a15 , typename util::add_rvalue_reference<A16>::type a16 , typename util::add_rvalue_reference<A17>::type a17 , typename util::add_rvalue_reference<A18>::type a18 , typename util::add_rvalue_reference<A19>::type a19 , typename util::add_rvalue_reference<A20>::type a20 , typename util::add_rvalue_reference<A21>::type a21)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20 , typename A21 , typename A22
      , typename IArchive
      , typename OArchive
    >
    struct empty_vtable<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21 , A22)
      , IArchive
      , OArchive
    >
        : empty_vtable_base
    {
        typedef R (*functor_type)(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21 , A22);
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21 , A22)
          , IArchive
          , OArchive
        > *get_ptr()
        {
            return
                get_empty_table<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21 , A22)
                >::template get<IArchive, OArchive>();
        }
        BOOST_ATTRIBUTE_NORETURN static R
        invoke(void ** f
            , typename util::add_rvalue_reference<A0>::type a0 , typename util::add_rvalue_reference<A1>::type a1 , typename util::add_rvalue_reference<A2>::type a2 , typename util::add_rvalue_reference<A3>::type a3 , typename util::add_rvalue_reference<A4>::type a4 , typename util::add_rvalue_reference<A5>::type a5 , typename util::add_rvalue_reference<A6>::type a6 , typename util::add_rvalue_reference<A7>::type a7 , typename util::add_rvalue_reference<A8>::type a8 , typename util::add_rvalue_reference<A9>::type a9 , typename util::add_rvalue_reference<A10>::type a10 , typename util::add_rvalue_reference<A11>::type a11 , typename util::add_rvalue_reference<A12>::type a12 , typename util::add_rvalue_reference<A13>::type a13 , typename util::add_rvalue_reference<A14>::type a14 , typename util::add_rvalue_reference<A15>::type a15 , typename util::add_rvalue_reference<A16>::type a16 , typename util::add_rvalue_reference<A17>::type a17 , typename util::add_rvalue_reference<A18>::type a18 , typename util::add_rvalue_reference<A19>::type a19 , typename util::add_rvalue_reference<A20>::type a20 , typename util::add_rvalue_reference<A21>::type a21 , typename util::add_rvalue_reference<A22>::type a22)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
