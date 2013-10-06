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
            , typename util::add_rvalue_reference<A0>::type);
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
            , typename util::add_rvalue_reference<A0>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type , typename util::add_rvalue_reference<A11>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type , typename util::add_rvalue_reference<A11>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type , typename util::add_rvalue_reference<A11>::type , typename util::add_rvalue_reference<A12>::type);
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
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type , typename util::add_rvalue_reference<A11>::type , typename util::add_rvalue_reference<A12>::type);
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type , typename util::add_rvalue_reference<A11>::type , typename util::add_rvalue_reference<A12>::type , typename util::add_rvalue_reference<A13>::type);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type , typename util::add_rvalue_reference<A11>::type , typename util::add_rvalue_reference<A12>::type , typename util::add_rvalue_reference<A13>::type);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type , typename util::add_rvalue_reference<A11>::type , typename util::add_rvalue_reference<A12>::type , typename util::add_rvalue_reference<A13>::type , typename util::add_rvalue_reference<A14>::type);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type , typename util::add_rvalue_reference<A11>::type , typename util::add_rvalue_reference<A12>::type , typename util::add_rvalue_reference<A13>::type , typename util::add_rvalue_reference<A14>::type);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type , typename util::add_rvalue_reference<A11>::type , typename util::add_rvalue_reference<A12>::type , typename util::add_rvalue_reference<A13>::type , typename util::add_rvalue_reference<A14>::type , typename util::add_rvalue_reference<A15>::type);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type , typename util::add_rvalue_reference<A11>::type , typename util::add_rvalue_reference<A12>::type , typename util::add_rvalue_reference<A13>::type , typename util::add_rvalue_reference<A14>::type , typename util::add_rvalue_reference<A15>::type);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type , typename util::add_rvalue_reference<A11>::type , typename util::add_rvalue_reference<A12>::type , typename util::add_rvalue_reference<A13>::type , typename util::add_rvalue_reference<A14>::type , typename util::add_rvalue_reference<A15>::type , typename util::add_rvalue_reference<A16>::type);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type , typename util::add_rvalue_reference<A11>::type , typename util::add_rvalue_reference<A12>::type , typename util::add_rvalue_reference<A13>::type , typename util::add_rvalue_reference<A14>::type , typename util::add_rvalue_reference<A15>::type , typename util::add_rvalue_reference<A16>::type);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type , typename util::add_rvalue_reference<A11>::type , typename util::add_rvalue_reference<A12>::type , typename util::add_rvalue_reference<A13>::type , typename util::add_rvalue_reference<A14>::type , typename util::add_rvalue_reference<A15>::type , typename util::add_rvalue_reference<A16>::type , typename util::add_rvalue_reference<A17>::type);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type , typename util::add_rvalue_reference<A11>::type , typename util::add_rvalue_reference<A12>::type , typename util::add_rvalue_reference<A13>::type , typename util::add_rvalue_reference<A14>::type , typename util::add_rvalue_reference<A15>::type , typename util::add_rvalue_reference<A16>::type , typename util::add_rvalue_reference<A17>::type);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type , typename util::add_rvalue_reference<A11>::type , typename util::add_rvalue_reference<A12>::type , typename util::add_rvalue_reference<A13>::type , typename util::add_rvalue_reference<A14>::type , typename util::add_rvalue_reference<A15>::type , typename util::add_rvalue_reference<A16>::type , typename util::add_rvalue_reference<A17>::type , typename util::add_rvalue_reference<A18>::type);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type , typename util::add_rvalue_reference<A11>::type , typename util::add_rvalue_reference<A12>::type , typename util::add_rvalue_reference<A13>::type , typename util::add_rvalue_reference<A14>::type , typename util::add_rvalue_reference<A15>::type , typename util::add_rvalue_reference<A16>::type , typename util::add_rvalue_reference<A17>::type , typename util::add_rvalue_reference<A18>::type);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
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
        : vtable_ptr_virtbase<IArchive, OArchive>
    {
        virtual ~vtable_ptr_base() {}
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void ** 
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type , typename util::add_rvalue_reference<A11>::type , typename util::add_rvalue_reference<A12>::type , typename util::add_rvalue_reference<A13>::type , typename util::add_rvalue_reference<A14>::type , typename util::add_rvalue_reference<A15>::type , typename util::add_rvalue_reference<A16>::type , typename util::add_rvalue_reference<A17>::type , typename util::add_rvalue_reference<A18>::type , typename util::add_rvalue_reference<A19>::type);
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
        virtual bool empty() const = 0;
        std::type_info const& (*get_type)();
        void (*static_delete)(void**);
        void (*destruct)(void**);
        void (*clone)(void * const*, void **);
        void (*copy)(void * const*, void **);
        R (*invoke)(void **
            , typename util::add_rvalue_reference<A0>::type , typename util::add_rvalue_reference<A1>::type , typename util::add_rvalue_reference<A2>::type , typename util::add_rvalue_reference<A3>::type , typename util::add_rvalue_reference<A4>::type , typename util::add_rvalue_reference<A5>::type , typename util::add_rvalue_reference<A6>::type , typename util::add_rvalue_reference<A7>::type , typename util::add_rvalue_reference<A8>::type , typename util::add_rvalue_reference<A9>::type , typename util::add_rvalue_reference<A10>::type , typename util::add_rvalue_reference<A11>::type , typename util::add_rvalue_reference<A12>::type , typename util::add_rvalue_reference<A13>::type , typename util::add_rvalue_reference<A14>::type , typename util::add_rvalue_reference<A15>::type , typename util::add_rvalue_reference<A16>::type , typename util::add_rvalue_reference<A17>::type , typename util::add_rvalue_reference<A18>::type , typename util::add_rvalue_reference<A19>::type);
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19
      , typename IArchive
      , typename OArchive
    >
    struct tracking_level<hpx::util::detail::vtable_ptr_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19), IArchive, OArchive
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
