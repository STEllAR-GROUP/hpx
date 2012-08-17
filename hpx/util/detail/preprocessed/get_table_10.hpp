// Copyright (c) 2007-2012 Hartmut Kaiser
// Copyright (c)      2012 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace util { namespace detail {
    template <
        typename Functor
      , typename R
       
    >
    struct get_table<
        Functor
      , R()
    >
    {
        template <typename IArchive, typename OArchive>
        static vtable_ptr_base<
            R()
          , IArchive
          , OArchive
        >*
        get()
        {
            typedef
                typename vtable<sizeof(Functor) <= sizeof(void *)>::
                    template type<
                        Functor
                      , R()
                      , IArchive
                      , OArchive
                    >
                vtable_type;
            typedef
                vtable_ptr<
                    R()
                  , IArchive
                  , OArchive
                  , vtable_type
                >
                vtable_ptr_type;
            static vtable_ptr_type ptr;
            return &ptr;
        }
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename Functor
      , typename R
      , typename A0
    >
    struct get_table<
        Functor
      , R(A0)
    >
    {
        template <typename IArchive, typename OArchive>
        static vtable_ptr_base<
            R(A0)
          , IArchive
          , OArchive
        >*
        get()
        {
            typedef
                typename vtable<sizeof(Functor) <= sizeof(void *)>::
                    template type<
                        Functor
                      , R(A0)
                      , IArchive
                      , OArchive
                    >
                vtable_type;
            typedef
                vtable_ptr<
                    R(A0)
                  , IArchive
                  , OArchive
                  , vtable_type
                >
                vtable_ptr_type;
            static vtable_ptr_type ptr;
            return &ptr;
        }
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename Functor
      , typename R
      , typename A0 , typename A1
    >
    struct get_table<
        Functor
      , R(A0 , A1)
    >
    {
        template <typename IArchive, typename OArchive>
        static vtable_ptr_base<
            R(A0 , A1)
          , IArchive
          , OArchive
        >*
        get()
        {
            typedef
                typename vtable<sizeof(Functor) <= sizeof(void *)>::
                    template type<
                        Functor
                      , R(A0 , A1)
                      , IArchive
                      , OArchive
                    >
                vtable_type;
            typedef
                vtable_ptr<
                    R(A0 , A1)
                  , IArchive
                  , OArchive
                  , vtable_type
                >
                vtable_ptr_type;
            static vtable_ptr_type ptr;
            return &ptr;
        }
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename Functor
      , typename R
      , typename A0 , typename A1 , typename A2
    >
    struct get_table<
        Functor
      , R(A0 , A1 , A2)
    >
    {
        template <typename IArchive, typename OArchive>
        static vtable_ptr_base<
            R(A0 , A1 , A2)
          , IArchive
          , OArchive
        >*
        get()
        {
            typedef
                typename vtable<sizeof(Functor) <= sizeof(void *)>::
                    template type<
                        Functor
                      , R(A0 , A1 , A2)
                      , IArchive
                      , OArchive
                    >
                vtable_type;
            typedef
                vtable_ptr<
                    R(A0 , A1 , A2)
                  , IArchive
                  , OArchive
                  , vtable_type
                >
                vtable_ptr_type;
            static vtable_ptr_type ptr;
            return &ptr;
        }
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename Functor
      , typename R
      , typename A0 , typename A1 , typename A2 , typename A3
    >
    struct get_table<
        Functor
      , R(A0 , A1 , A2 , A3)
    >
    {
        template <typename IArchive, typename OArchive>
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3)
          , IArchive
          , OArchive
        >*
        get()
        {
            typedef
                typename vtable<sizeof(Functor) <= sizeof(void *)>::
                    template type<
                        Functor
                      , R(A0 , A1 , A2 , A3)
                      , IArchive
                      , OArchive
                    >
                vtable_type;
            typedef
                vtable_ptr<
                    R(A0 , A1 , A2 , A3)
                  , IArchive
                  , OArchive
                  , vtable_type
                >
                vtable_ptr_type;
            static vtable_ptr_type ptr;
            return &ptr;
        }
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename Functor
      , typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
    >
    struct get_table<
        Functor
      , R(A0 , A1 , A2 , A3 , A4)
    >
    {
        template <typename IArchive, typename OArchive>
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4)
          , IArchive
          , OArchive
        >*
        get()
        {
            typedef
                typename vtable<sizeof(Functor) <= sizeof(void *)>::
                    template type<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4)
                      , IArchive
                      , OArchive
                    >
                vtable_type;
            typedef
                vtable_ptr<
                    R(A0 , A1 , A2 , A3 , A4)
                  , IArchive
                  , OArchive
                  , vtable_type
                >
                vtable_ptr_type;
            static vtable_ptr_type ptr;
            return &ptr;
        }
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename Functor
      , typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
    >
    struct get_table<
        Functor
      , R(A0 , A1 , A2 , A3 , A4 , A5)
    >
    {
        template <typename IArchive, typename OArchive>
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5)
          , IArchive
          , OArchive
        >*
        get()
        {
            typedef
                typename vtable<sizeof(Functor) <= sizeof(void *)>::
                    template type<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5)
                      , IArchive
                      , OArchive
                    >
                vtable_type;
            typedef
                vtable_ptr<
                    R(A0 , A1 , A2 , A3 , A4 , A5)
                  , IArchive
                  , OArchive
                  , vtable_type
                >
                vtable_ptr_type;
            static vtable_ptr_type ptr;
            return &ptr;
        }
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename Functor
      , typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
    >
    struct get_table<
        Functor
      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
    >
    {
        template <typename IArchive, typename OArchive>
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
          , IArchive
          , OArchive
        >*
        get()
        {
            typedef
                typename vtable<sizeof(Functor) <= sizeof(void *)>::
                    template type<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
                      , IArchive
                      , OArchive
                    >
                vtable_type;
            typedef
                vtable_ptr<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
                  , IArchive
                  , OArchive
                  , vtable_type
                >
                vtable_ptr_type;
            static vtable_ptr_type ptr;
            return &ptr;
        }
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename Functor
      , typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
    >
    struct get_table<
        Functor
      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
    >
    {
        template <typename IArchive, typename OArchive>
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
          , IArchive
          , OArchive
        >*
        get()
        {
            typedef
                typename vtable<sizeof(Functor) <= sizeof(void *)>::
                    template type<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
                      , IArchive
                      , OArchive
                    >
                vtable_type;
            typedef
                vtable_ptr<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
                  , IArchive
                  , OArchive
                  , vtable_type
                >
                vtable_ptr_type;
            static vtable_ptr_type ptr;
            return &ptr;
        }
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename Functor
      , typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
    >
    struct get_table<
        Functor
      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
    >
    {
        template <typename IArchive, typename OArchive>
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
          , IArchive
          , OArchive
        >*
        get()
        {
            typedef
                typename vtable<sizeof(Functor) <= sizeof(void *)>::
                    template type<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
                      , IArchive
                      , OArchive
                    >
                vtable_type;
            typedef
                vtable_ptr<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
                  , IArchive
                  , OArchive
                  , vtable_type
                >
                vtable_ptr_type;
            static vtable_ptr_type ptr;
            return &ptr;
        }
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename Functor
      , typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
    >
    struct get_table<
        Functor
      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
    >
    {
        template <typename IArchive, typename OArchive>
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
          , IArchive
          , OArchive
        >*
        get()
        {
            typedef
                typename vtable<sizeof(Functor) <= sizeof(void *)>::
                    template type<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
                      , IArchive
                      , OArchive
                    >
                vtable_type;
            typedef
                vtable_ptr<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
                  , IArchive
                  , OArchive
                  , vtable_type
                >
                vtable_ptr_type;
            static vtable_ptr_type ptr;
            return &ptr;
        }
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename Functor
      , typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
    >
    struct get_table<
        Functor
      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
    >
    {
        template <typename IArchive, typename OArchive>
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
          , IArchive
          , OArchive
        >*
        get()
        {
            typedef
                typename vtable<sizeof(Functor) <= sizeof(void *)>::
                    template type<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
                      , IArchive
                      , OArchive
                    >
                vtable_type;
            typedef
                vtable_ptr<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
                  , IArchive
                  , OArchive
                  , vtable_type
                >
                vtable_ptr_type;
            static vtable_ptr_type ptr;
            return &ptr;
        }
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename Functor
      , typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
    >
    struct get_table<
        Functor
      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
    >
    {
        template <typename IArchive, typename OArchive>
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
          , IArchive
          , OArchive
        >*
        get()
        {
            typedef
                typename vtable<sizeof(Functor) <= sizeof(void *)>::
                    template type<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
                      , IArchive
                      , OArchive
                    >
                vtable_type;
            typedef
                vtable_ptr<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
                  , IArchive
                  , OArchive
                  , vtable_type
                >
                vtable_ptr_type;
            static vtable_ptr_type ptr;
            return &ptr;
        }
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename Functor
      , typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
    >
    struct get_table<
        Functor
      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
    >
    {
        template <typename IArchive, typename OArchive>
        static vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
          , IArchive
          , OArchive
        >*
        get()
        {
            typedef
                typename vtable<sizeof(Functor) <= sizeof(void *)>::
                    template type<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
                      , IArchive
                      , OArchive
                    >
                vtable_type;
            typedef
                vtable_ptr<
                    R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
                  , IArchive
                  , OArchive
                  , vtable_type
                >
                vtable_ptr_type;
            static vtable_ptr_type ptr;
            return &ptr;
        }
    };
}}}
