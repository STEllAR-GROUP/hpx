// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace util { namespace detail {
    
    template <typename Function>
    struct init_registration;
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
        : util::detail::vtable_ptr_base<
            R()
          , IArchive
          , OArchive
        >
    {
        typedef
            util::detail::vtable_ptr_virtbase<IArchive, OArchive>
            vtable_ptr_virtbase_type;
        typedef
            util::detail::vtable_ptr_base<
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
        }
        ~vtable_ptr()
        {
            init_registration<vtable_ptr>::g.register_function();
        }
        char const* get_function_name() const
        {
            return util::detail::get_function_name<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar << Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar >> Vtable::construct(object);
        }
    };
    
    
    template <
        typename R
      
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct init_registration<
        vtable_ptr<
            R()
          , IArchive
          , OArchive
          , Vtable
        >
    >
    {
        typedef vtable_ptr<
            R()
          , IArchive
          , OArchive
          , Vtable
        > vtable_ptr_type;
        static automatic_function_registration<vtable_ptr_type> g;
    };
    template <
        typename R
      
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    automatic_function_registration<
        vtable_ptr<
            R()
          , IArchive
          , OArchive
          , Vtable
        >
    >
        init_registration<
            vtable_ptr<
                R()
              , IArchive
              , OArchive
              , Vtable
            >
        >::g = automatic_function_registration<
                    vtable_ptr<
                        R()
                      , IArchive
                      , OArchive
                      , Vtable
                    >
                >();
    
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
        : util::detail::vtable_ptr_base<
            R()
          , void
          , void
        >
    {
        typedef
            util::detail::vtable_ptr_base<
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
    
    template <typename Function>
    struct init_registration;
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
        : util::detail::vtable_ptr_base<
            R(A0)
          , IArchive
          , OArchive
        >
    {
        typedef
            util::detail::vtable_ptr_virtbase<IArchive, OArchive>
            vtable_ptr_virtbase_type;
        typedef
            util::detail::vtable_ptr_base<
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
        }
        ~vtable_ptr()
        {
            init_registration<vtable_ptr>::g.register_function();
        }
        char const* get_function_name() const
        {
            return util::detail::get_function_name<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar << Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar >> Vtable::construct(object);
        }
    };
    
    
    template <
        typename R
      , typename A0
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct init_registration<
        vtable_ptr<
            R(A0)
          , IArchive
          , OArchive
          , Vtable
        >
    >
    {
        typedef vtable_ptr<
            R(A0)
          , IArchive
          , OArchive
          , Vtable
        > vtable_ptr_type;
        static automatic_function_registration<vtable_ptr_type> g;
    };
    template <
        typename R
      , typename A0
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    automatic_function_registration<
        vtable_ptr<
            R(A0)
          , IArchive
          , OArchive
          , Vtable
        >
    >
        init_registration<
            vtable_ptr<
                R(A0)
              , IArchive
              , OArchive
              , Vtable
            >
        >::g = automatic_function_registration<
                    vtable_ptr<
                        R(A0)
                      , IArchive
                      , OArchive
                      , Vtable
                    >
                >();
    
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
        : util::detail::vtable_ptr_base<
            R(A0)
          , void
          , void
        >
    {
        typedef
            util::detail::vtable_ptr_base<
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
    
    template <typename Function>
    struct init_registration;
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
        : util::detail::vtable_ptr_base<
            R(A0 , A1)
          , IArchive
          , OArchive
        >
    {
        typedef
            util::detail::vtable_ptr_virtbase<IArchive, OArchive>
            vtable_ptr_virtbase_type;
        typedef
            util::detail::vtable_ptr_base<
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
        }
        ~vtable_ptr()
        {
            init_registration<vtable_ptr>::g.register_function();
        }
        char const* get_function_name() const
        {
            return util::detail::get_function_name<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar << Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar >> Vtable::construct(object);
        }
    };
    
    
    template <
        typename R
      , typename A0 , typename A1
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct init_registration<
        vtable_ptr<
            R(A0 , A1)
          , IArchive
          , OArchive
          , Vtable
        >
    >
    {
        typedef vtable_ptr<
            R(A0 , A1)
          , IArchive
          , OArchive
          , Vtable
        > vtable_ptr_type;
        static automatic_function_registration<vtable_ptr_type> g;
    };
    template <
        typename R
      , typename A0 , typename A1
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    automatic_function_registration<
        vtable_ptr<
            R(A0 , A1)
          , IArchive
          , OArchive
          , Vtable
        >
    >
        init_registration<
            vtable_ptr<
                R(A0 , A1)
              , IArchive
              , OArchive
              , Vtable
            >
        >::g = automatic_function_registration<
                    vtable_ptr<
                        R(A0 , A1)
                      , IArchive
                      , OArchive
                      , Vtable
                    >
                >();
    
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
        : util::detail::vtable_ptr_base<
            R(A0 , A1)
          , void
          , void
        >
    {
        typedef
            util::detail::vtable_ptr_base<
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
    
    template <typename Function>
    struct init_registration;
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
        : util::detail::vtable_ptr_base<
            R(A0 , A1 , A2)
          , IArchive
          , OArchive
        >
    {
        typedef
            util::detail::vtable_ptr_virtbase<IArchive, OArchive>
            vtable_ptr_virtbase_type;
        typedef
            util::detail::vtable_ptr_base<
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
        }
        ~vtable_ptr()
        {
            init_registration<vtable_ptr>::g.register_function();
        }
        char const* get_function_name() const
        {
            return util::detail::get_function_name<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar << Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar >> Vtable::construct(object);
        }
    };
    
    
    template <
        typename R
      , typename A0 , typename A1 , typename A2
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct init_registration<
        vtable_ptr<
            R(A0 , A1 , A2)
          , IArchive
          , OArchive
          , Vtable
        >
    >
    {
        typedef vtable_ptr<
            R(A0 , A1 , A2)
          , IArchive
          , OArchive
          , Vtable
        > vtable_ptr_type;
        static automatic_function_registration<vtable_ptr_type> g;
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    automatic_function_registration<
        vtable_ptr<
            R(A0 , A1 , A2)
          , IArchive
          , OArchive
          , Vtable
        >
    >
        init_registration<
            vtable_ptr<
                R(A0 , A1 , A2)
              , IArchive
              , OArchive
              , Vtable
            >
        >::g = automatic_function_registration<
                    vtable_ptr<
                        R(A0 , A1 , A2)
                      , IArchive
                      , OArchive
                      , Vtable
                    >
                >();
    
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
        : util::detail::vtable_ptr_base<
            R(A0 , A1 , A2)
          , void
          , void
        >
    {
        typedef
            util::detail::vtable_ptr_base<
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
    
    template <typename Function>
    struct init_registration;
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
        : util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3)
          , IArchive
          , OArchive
        >
    {
        typedef
            util::detail::vtable_ptr_virtbase<IArchive, OArchive>
            vtable_ptr_virtbase_type;
        typedef
            util::detail::vtable_ptr_base<
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
        }
        ~vtable_ptr()
        {
            init_registration<vtable_ptr>::g.register_function();
        }
        char const* get_function_name() const
        {
            return util::detail::get_function_name<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar << Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar >> Vtable::construct(object);
        }
    };
    
    
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct init_registration<
        vtable_ptr<
            R(A0 , A1 , A2 , A3)
          , IArchive
          , OArchive
          , Vtable
        >
    >
    {
        typedef vtable_ptr<
            R(A0 , A1 , A2 , A3)
          , IArchive
          , OArchive
          , Vtable
        > vtable_ptr_type;
        static automatic_function_registration<vtable_ptr_type> g;
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    automatic_function_registration<
        vtable_ptr<
            R(A0 , A1 , A2 , A3)
          , IArchive
          , OArchive
          , Vtable
        >
    >
        init_registration<
            vtable_ptr<
                R(A0 , A1 , A2 , A3)
              , IArchive
              , OArchive
              , Vtable
            >
        >::g = automatic_function_registration<
                    vtable_ptr<
                        R(A0 , A1 , A2 , A3)
                      , IArchive
                      , OArchive
                      , Vtable
                    >
                >();
    
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
        : util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3)
          , void
          , void
        >
    {
        typedef
            util::detail::vtable_ptr_base<
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
    
    template <typename Function>
    struct init_registration;
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
        : util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4)
          , IArchive
          , OArchive
        >
    {
        typedef
            util::detail::vtable_ptr_virtbase<IArchive, OArchive>
            vtable_ptr_virtbase_type;
        typedef
            util::detail::vtable_ptr_base<
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
        }
        ~vtable_ptr()
        {
            init_registration<vtable_ptr>::g.register_function();
        }
        char const* get_function_name() const
        {
            return util::detail::get_function_name<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar << Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar >> Vtable::construct(object);
        }
    };
    
    
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct init_registration<
        vtable_ptr<
            R(A0 , A1 , A2 , A3 , A4)
          , IArchive
          , OArchive
          , Vtable
        >
    >
    {
        typedef vtable_ptr<
            R(A0 , A1 , A2 , A3 , A4)
          , IArchive
          , OArchive
          , Vtable
        > vtable_ptr_type;
        static automatic_function_registration<vtable_ptr_type> g;
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    automatic_function_registration<
        vtable_ptr<
            R(A0 , A1 , A2 , A3 , A4)
          , IArchive
          , OArchive
          , Vtable
        >
    >
        init_registration<
            vtable_ptr<
                R(A0 , A1 , A2 , A3 , A4)
              , IArchive
              , OArchive
              , Vtable
            >
        >::g = automatic_function_registration<
                    vtable_ptr<
                        R(A0 , A1 , A2 , A3 , A4)
                      , IArchive
                      , OArchive
                      , Vtable
                    >
                >();
    
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
        : util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4)
          , void
          , void
        >
    {
        typedef
            util::detail::vtable_ptr_base<
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
    
    template <typename Function>
    struct init_registration;
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
        : util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5)
          , IArchive
          , OArchive
        >
    {
        typedef
            util::detail::vtable_ptr_virtbase<IArchive, OArchive>
            vtable_ptr_virtbase_type;
        typedef
            util::detail::vtable_ptr_base<
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
        }
        ~vtable_ptr()
        {
            init_registration<vtable_ptr>::g.register_function();
        }
        char const* get_function_name() const
        {
            return util::detail::get_function_name<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar << Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar >> Vtable::construct(object);
        }
    };
    
    
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct init_registration<
        vtable_ptr<
            R(A0 , A1 , A2 , A3 , A4 , A5)
          , IArchive
          , OArchive
          , Vtable
        >
    >
    {
        typedef vtable_ptr<
            R(A0 , A1 , A2 , A3 , A4 , A5)
          , IArchive
          , OArchive
          , Vtable
        > vtable_ptr_type;
        static automatic_function_registration<vtable_ptr_type> g;
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    automatic_function_registration<
        vtable_ptr<
            R(A0 , A1 , A2 , A3 , A4 , A5)
          , IArchive
          , OArchive
          , Vtable
        >
    >
        init_registration<
            vtable_ptr<
                R(A0 , A1 , A2 , A3 , A4 , A5)
              , IArchive
              , OArchive
              , Vtable
            >
        >::g = automatic_function_registration<
                    vtable_ptr<
                        R(A0 , A1 , A2 , A3 , A4 , A5)
                      , IArchive
                      , OArchive
                      , Vtable
                    >
                >();
    
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
        : util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5)
          , void
          , void
        >
    {
        typedef
            util::detail::vtable_ptr_base<
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
    
    template <typename Function>
    struct init_registration;
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
        : util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
          , IArchive
          , OArchive
        >
    {
        typedef
            util::detail::vtable_ptr_virtbase<IArchive, OArchive>
            vtable_ptr_virtbase_type;
        typedef
            util::detail::vtable_ptr_base<
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
        }
        ~vtable_ptr()
        {
            init_registration<vtable_ptr>::g.register_function();
        }
        char const* get_function_name() const
        {
            return util::detail::get_function_name<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar << Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar >> Vtable::construct(object);
        }
    };
    
    
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct init_registration<
        vtable_ptr<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
          , IArchive
          , OArchive
          , Vtable
        >
    >
    {
        typedef vtable_ptr<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
          , IArchive
          , OArchive
          , Vtable
        > vtable_ptr_type;
        static automatic_function_registration<vtable_ptr_type> g;
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    automatic_function_registration<
        vtable_ptr<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
          , IArchive
          , OArchive
          , Vtable
        >
    >
        init_registration<
            vtable_ptr<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
              , IArchive
              , OArchive
              , Vtable
            >
        >::g = automatic_function_registration<
                    vtable_ptr<
                        R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
                      , IArchive
                      , OArchive
                      , Vtable
                    >
                >();
    
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
        : util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
          , void
          , void
        >
    {
        typedef
            util::detail::vtable_ptr_base<
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
    
    template <typename Function>
    struct init_registration;
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
        : util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
          , IArchive
          , OArchive
        >
    {
        typedef
            util::detail::vtable_ptr_virtbase<IArchive, OArchive>
            vtable_ptr_virtbase_type;
        typedef
            util::detail::vtable_ptr_base<
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
        }
        ~vtable_ptr()
        {
            init_registration<vtable_ptr>::g.register_function();
        }
        char const* get_function_name() const
        {
            return util::detail::get_function_name<vtable_ptr>();
        }
        virtual bool empty() const
        {
            return Vtable::empty;
        }
        virtual base_type * get_ptr()
        {
            return Vtable::get_ptr();
        }
        void save_object(void *const* object, OArchive & ar, unsigned)
        {
            ar << Vtable::get(object);
        }
        void load_object(void ** object, IArchive & ar, unsigned)
        {
            ar >> Vtable::construct(object);
        }
    };
    
    
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    struct init_registration<
        vtable_ptr<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
          , IArchive
          , OArchive
          , Vtable
        >
    >
    {
        typedef vtable_ptr<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
          , IArchive
          , OArchive
          , Vtable
        > vtable_ptr_type;
        static automatic_function_registration<vtable_ptr_type> g;
    };
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename IArchive
      , typename OArchive
      , typename Vtable
    >
    automatic_function_registration<
        vtable_ptr<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
          , IArchive
          , OArchive
          , Vtable
        >
    >
        init_registration<
            vtable_ptr<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
              , IArchive
              , OArchive
              , Vtable
            >
        >::g = automatic_function_registration<
                    vtable_ptr<
                        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
                      , IArchive
                      , OArchive
                      , Vtable
                    >
                >();
    
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
        : util::detail::vtable_ptr_base<
            R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
          , void
          , void
        >
    {
        typedef
            util::detail::vtable_ptr_base<
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
