// Copyright (c) 2007-2012 Hartmut Kaiser
// Copyright (c)      2012 Thomas Heller
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
