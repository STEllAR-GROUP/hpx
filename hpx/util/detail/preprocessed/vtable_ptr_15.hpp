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
      , typename Vtable
    >
    struct vtable_ptr<
        R()
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R()
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R()
              , IArchive
              , OArchive
            >
            base_type;
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
            
            hpx::actions::detail::guid_initialization<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        static void register_base()
        {
            util::void_cast_register_nonvirt<vtable_ptr, base_type>();
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar & Vtable::construct(object);
        }
        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    template <
        typename R
       
      , typename Vtable
    >
    struct vtable_ptr<
        R()
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R()
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R()
              , void
              , void
            >
            base_type;
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
       
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(), IArchive, OArchive, Vtable
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
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0)
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0)
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0)
              , IArchive
              , OArchive
            >
            base_type;
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
            
            hpx::actions::detail::guid_initialization<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        static void register_base()
        {
            util::void_cast_register_nonvirt<vtable_ptr, base_type>();
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar & Vtable::construct(object);
        }
        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    template <
        typename R
      , typename A0
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0)
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0)
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0)
              , void
              , void
            >
            base_type;
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(A0), IArchive, OArchive, Vtable
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
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1)
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1)
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1)
              , IArchive
              , OArchive
            >
            base_type;
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
            
            hpx::actions::detail::guid_initialization<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        static void register_base()
        {
            util::void_cast_register_nonvirt<vtable_ptr, base_type>();
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar & Vtable::construct(object);
        }
        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    template <
        typename R
      , typename A0 , typename A1
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1)
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1)
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1)
              , void
              , void
            >
            base_type;
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(A0 , A1), IArchive, OArchive, Vtable
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
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2)
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2)
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2)
              , IArchive
              , OArchive
            >
            base_type;
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
            
            hpx::actions::detail::guid_initialization<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        static void register_base()
        {
            util::void_cast_register_nonvirt<vtable_ptr, base_type>();
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar & Vtable::construct(object);
        }
        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2)
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2)
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2)
              , void
              , void
            >
            base_type;
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(A0 , A1 , A2), IArchive, OArchive, Vtable
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
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3)
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3)
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3)
              , IArchive
              , OArchive
            >
            base_type;
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
            
            hpx::actions::detail::guid_initialization<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        static void register_base()
        {
            util::void_cast_register_nonvirt<vtable_ptr, base_type>();
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar & Vtable::construct(object);
        }
        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3)
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3)
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3)
              , void
              , void
            >
            base_type;
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(A0 , A1 , A2 , A3), IArchive, OArchive, Vtable
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
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4)
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4)
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4)
              , IArchive
              , OArchive
            >
            base_type;
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
            
            hpx::actions::detail::guid_initialization<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        static void register_base()
        {
            util::void_cast_register_nonvirt<vtable_ptr, base_type>();
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar & Vtable::construct(object);
        }
        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4)
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4)
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4)
              , void
              , void
            >
            base_type;
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4), IArchive, OArchive, Vtable
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
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5)
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5)
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5)
              , IArchive
              , OArchive
            >
            base_type;
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
            
            hpx::actions::detail::guid_initialization<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        static void register_base()
        {
            util::void_cast_register_nonvirt<vtable_ptr, base_type>();
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar & Vtable::construct(object);
        }
        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5)
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5)
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5)
              , void
              , void
            >
            base_type;
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5), IArchive, OArchive, Vtable
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
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
              , IArchive
              , OArchive
            >
            base_type;
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
            
            hpx::actions::detail::guid_initialization<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        static void register_base()
        {
            util::void_cast_register_nonvirt<vtable_ptr, base_type>();
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar & Vtable::construct(object);
        }
        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
              , void
              , void
            >
            base_type;
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6), IArchive, OArchive, Vtable
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
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
              , IArchive
              , OArchive
            >
            base_type;
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
            
            hpx::actions::detail::guid_initialization<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        static void register_base()
        {
            util::void_cast_register_nonvirt<vtable_ptr, base_type>();
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar & Vtable::construct(object);
        }
        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
              , void
              , void
            >
            base_type;
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7), IArchive, OArchive, Vtable
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
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
              , IArchive
              , OArchive
            >
            base_type;
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
            
            hpx::actions::detail::guid_initialization<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        static void register_base()
        {
            util::void_cast_register_nonvirt<vtable_ptr, base_type>();
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar & Vtable::construct(object);
        }
        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
              , void
              , void
            >
            base_type;
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8), IArchive, OArchive, Vtable
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
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
              , IArchive
              , OArchive
            >
            base_type;
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
            
            hpx::actions::detail::guid_initialization<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        static void register_base()
        {
            util::void_cast_register_nonvirt<vtable_ptr, base_type>();
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar & Vtable::construct(object);
        }
        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
              , void
              , void
            >
            base_type;
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9), IArchive, OArchive, Vtable
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
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
              , IArchive
              , OArchive
            >
            base_type;
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
            
            hpx::actions::detail::guid_initialization<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        static void register_base()
        {
            util::void_cast_register_nonvirt<vtable_ptr, base_type>();
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar & Vtable::construct(object);
        }
        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
              , void
              , void
            >
            base_type;
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10), IArchive, OArchive, Vtable
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
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
              , IArchive
              , OArchive
            >
            base_type;
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
            
            hpx::actions::detail::guid_initialization<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        static void register_base()
        {
            util::void_cast_register_nonvirt<vtable_ptr, base_type>();
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar & Vtable::construct(object);
        }
        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
              , void
              , void
            >
            base_type;
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11), IArchive, OArchive, Vtable
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
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
              , IArchive
              , OArchive
            >
            base_type;
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
            
            hpx::actions::detail::guid_initialization<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        static void register_base()
        {
            util::void_cast_register_nonvirt<vtable_ptr, base_type>();
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar & Vtable::construct(object);
        }
        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
              , void
              , void
            >
            base_type;
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12), IArchive, OArchive, Vtable
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
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
              , IArchive
              , OArchive
            >
            base_type;
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
            
            hpx::actions::detail::guid_initialization<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        static void register_base()
        {
            util::void_cast_register_nonvirt<vtable_ptr, base_type>();
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar & Vtable::construct(object);
        }
        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
              , void
              , void
            >
            base_type;
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13), IArchive, OArchive, Vtable
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
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
              , IArchive
              , OArchive
            >
            base_type;
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
            
            hpx::actions::detail::guid_initialization<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        static void register_base()
        {
            util::void_cast_register_nonvirt<vtable_ptr, base_type>();
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar & Vtable::construct(object);
        }
        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
              , void
              , void
            >
            base_type;
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14), IArchive, OArchive, Vtable
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
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
              , IArchive
              , OArchive
            >
            base_type;
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
            
            hpx::actions::detail::guid_initialization<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        static void register_base()
        {
            util::void_cast_register_nonvirt<vtable_ptr, base_type>();
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar & Vtable::construct(object);
        }
        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
              , void
              , void
            >
            base_type;
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15), IArchive, OArchive, Vtable
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
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
              , IArchive
              , OArchive
            >
            base_type;
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
            
            hpx::actions::detail::guid_initialization<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        static void register_base()
        {
            util::void_cast_register_nonvirt<vtable_ptr, base_type>();
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar & Vtable::construct(object);
        }
        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
              , void
              , void
            >
            base_type;
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16), IArchive, OArchive, Vtable
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
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
      , IArchive
      , OArchive
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
          , IArchive
          , OArchive
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
              , IArchive
              , OArchive
            >
            base_type;
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
            
            hpx::actions::detail::guid_initialization<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        static void register_base()
        {
            util::void_cast_register_nonvirt<vtable_ptr, base_type>();
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar & Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar & Vtable::construct(object);
        }
        template <typename Archive>
        BOOST_FORCEINLINE void serialize(Archive & ar, unsigned)
        {
            ar & boost::serialization::base_object<base_type>(*this);
        }
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17
      , typename Vtable
    >
    struct vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
      , void
      , void
      , Vtable
    >
        : hpx::util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
          , void
          , void
        >
    {
        typedef
            hpx::util::detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
              , void
              , void
            >
            base_type;
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        vtable_ptr()
        {
            base_type::get_type = Vtable::get_type;
            base_type::static_delete = Vtable::static_delete;
            base_type::destruct = Vtable::destruct;
            base_type::clone = Vtable::clone;
            base_type::copy = Vtable::copy;
            base_type::invoke = Vtable::invoke;
        }
    };
}}}
namespace boost { namespace serialization {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct tracking_level<hpx::util::detail::vtable_ptr<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17), IArchive, OArchive, Vtable
    > >
        : boost::mpl::int_<boost::serialization::track_never>
    {};
}}
