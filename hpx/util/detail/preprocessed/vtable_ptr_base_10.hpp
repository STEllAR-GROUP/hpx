// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            );
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            );
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , A0 &&);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , A0 &&);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , A0 && , A1 &&);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , A0 && , A1 &&);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0 , A1), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , A0 && , A1 && , A2 &&);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , A0 && , A1 && , A2 &&);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0 , A1 , A2), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , A0 && , A1 && , A2 && , A3 &&);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , A0 && , A1 && , A2 && , A3 &&);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0 , A1 , A2 , A3), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , A0 && , A1 && , A2 && , A3 && , A4 &&);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , A0 && , A1 && , A2 && , A3 && , A4 &&);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , A0 && , A1 && , A2 && , A3 && , A4 && , A5 &&);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , A0 && , A1 && , A2 && , A3 && , A4 && , A5 &&);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 &&);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 &&);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 &&);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 &&);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 &&);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 &&);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 &&);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 &&);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 && , A10 &&);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 && , A10 &&);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 && , A10 && , A11 &&);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 && , A10 && , A11 &&);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 && , A10 && , A11 && , A12 &&);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , A0 && , A1 && , A2 && , A3 && , A4 && , A5 && , A6 && , A7 && , A8 && , A9 && , A10 && , A11 && , A12 &&);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
