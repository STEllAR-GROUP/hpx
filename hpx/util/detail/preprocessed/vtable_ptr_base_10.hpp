// Copyright (c) 2007-2012 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace util { namespace detail {
    template <
        typename R
       
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R()
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** );
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
       
    >
    struct vtable_ptr_base<
        R()
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** );
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0
    >
    struct vtable_ptr_base<
        R(A0)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0);
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0 , typename A1
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0 , A1)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0 , typename A1
    >
    struct vtable_ptr_base<
        R(A0 , A1)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1);
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0 , typename A1 , typename A2
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2);
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3);
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4);
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5);
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6);
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7);
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8);
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9);
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10);
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11);
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12);
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13);
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14);
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15);
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16);
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17);
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18);
    };
}}}
namespace hpx { namespace util { namespace detail {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19
      , typename IArchive
      , typename OArchive
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19)
      , IArchive
      , OArchive
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19);
        virtual void save_object(void *const*, OArchive & ar, unsigned) = 0;
        virtual void load_object(void **, IArchive & ar, unsigned) = 0;
        template <typename Archive>
        void serialize(Archive & ar, unsigned)
        {}
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19
    >
    struct vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19)
      , void
      , void
    >
    {
        virtual ~vtable_ptr_base() {}
        virtual vtable_ptr_base * get_ptr() = 0;
        boost::detail::sp_typeinfo const & (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** , A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19);
    };
}}}
