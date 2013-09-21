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
            , typename util::detail::add_rvalue_reference<A0>::type a0)
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
            , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1)
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
            , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2)
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
            , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3)
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
            , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4)
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
            , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4 , typename util::detail::add_rvalue_reference<A5>::type a5)
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
            , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4 , typename util::detail::add_rvalue_reference<A5>::type a5 , typename util::detail::add_rvalue_reference<A6>::type a6)
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
            , typename util::detail::add_rvalue_reference<A0>::type a0 , typename util::detail::add_rvalue_reference<A1>::type a1 , typename util::detail::add_rvalue_reference<A2>::type a2 , typename util::detail::add_rvalue_reference<A3>::type a3 , typename util::detail::add_rvalue_reference<A4>::type a4 , typename util::detail::add_rvalue_reference<A5>::type a5 , typename util::detail::add_rvalue_reference<A6>::type a6 , typename util::detail::add_rvalue_reference<A7>::type a7)
        {
            hpx::throw_exception(bad_function_call,
                "empty function object should not be used",
                "empty_vtable::operator()");
        }
    };
}}}
