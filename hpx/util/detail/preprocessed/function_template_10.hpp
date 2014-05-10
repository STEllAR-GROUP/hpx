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
      
      , typename IArchive, typename OArchive
    >
    struct function_base<
        R()
      , IArchive, OArchive
    >
    {
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R()
              , IArchive, OArchive
            > vtable_ptr_type;
        
        function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * = 0
        ) : vptr(get_empty_table_ptr())
          , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename util::decay<Functor>::type
                    functor_type;
                vptr = get_table_ptr<functor_type>();
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
          : vptr(get_empty_table_ptr())
          , object(0)
        {
            assign(other);
        }
        function_base(function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        function_base &assign(function_base const & other)
        {
            if(&other != this)
            {
                if(vptr == other.vptr && !empty())
                {
                    vptr->copy(&other.object, &object);
                }
                else
                {
                    reset();
                    if(!other.empty())
                    {
                        other.vptr->clone(&other.object, &object);
                        vptr = other.vptr;
                    }
                }
            }
            return *this;
        }
        template <typename Functor>
        function_base & assign(Functor && f)
        {
            if (this == &f)
                return *this;
            typedef
                typename util::decay<Functor>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if(vptr == f_vptr && !empty())
            {
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    vptr->destruct(&object);
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(std::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(std::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        function_base & operator=(function_base const & t)
        {
            return assign(t);
        }
        function_base & operator=(function_base && t) BOOST_NOEXCEPT
        {
            if(this != &t)
            {
                reset();
                vptr = t.vptr;
                object = t.object;
                t.vptr = get_empty_table_ptr();
                t.object = 0;
            }
            return *this;
        }
        function_base &swap(function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const BOOST_NOEXCEPT
        {
            return empty();
        }
        void reset() BOOST_NOEXCEPT
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_empty_table<
                        R()
                    >::template get<IArchive, OArchive>();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R()
                    >::template get<IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()() const
        {
            return vptr->invoke(&object
                 );
        }
        std::type_info const& target_type() const BOOST_NOEXCEPT
        {
            return vptr->get_type();
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type *>(&object);
            return reinterpret_cast<functor_type *>(object);
        }
        template <typename T>
        T const* target() const BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type const*>(&object);
            return reinterpret_cast<functor_type const*>(object);
        }
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0
      , typename IArchive, typename OArchive
    >
    struct function_base<
        R(A0)
      , IArchive, OArchive
    >
    {
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0)
              , IArchive, OArchive
            > vtable_ptr_type;
        
        function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * = 0
        ) : vptr(get_empty_table_ptr())
          , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename util::decay<Functor>::type
                    functor_type;
                vptr = get_table_ptr<functor_type>();
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
          : vptr(get_empty_table_ptr())
          , object(0)
        {
            assign(other);
        }
        function_base(function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        function_base &assign(function_base const & other)
        {
            if(&other != this)
            {
                if(vptr == other.vptr && !empty())
                {
                    vptr->copy(&other.object, &object);
                }
                else
                {
                    reset();
                    if(!other.empty())
                    {
                        other.vptr->clone(&other.object, &object);
                        vptr = other.vptr;
                    }
                }
            }
            return *this;
        }
        template <typename Functor>
        function_base & assign(Functor && f)
        {
            if (this == &f)
                return *this;
            typedef
                typename util::decay<Functor>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if(vptr == f_vptr && !empty())
            {
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    vptr->destruct(&object);
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(std::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(std::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        function_base & operator=(function_base const & t)
        {
            return assign(t);
        }
        function_base & operator=(function_base && t) BOOST_NOEXCEPT
        {
            if(this != &t)
            {
                reset();
                vptr = t.vptr;
                object = t.object;
                t.vptr = get_empty_table_ptr();
                t.object = 0;
            }
            return *this;
        }
        function_base &swap(function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const BOOST_NOEXCEPT
        {
            return empty();
        }
        void reset() BOOST_NOEXCEPT
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_empty_table<
                        R(A0)
                    >::template get<IArchive, OArchive>();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R(A0)
                    >::template get<IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0) const
        {
            return vptr->invoke(&object
                , std::forward<A0>( a0 ));
        }
        std::type_info const& target_type() const BOOST_NOEXCEPT
        {
            return vptr->get_type();
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type *>(&object);
            return reinterpret_cast<functor_type *>(object);
        }
        template <typename T>
        T const* target() const BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type const*>(&object);
            return reinterpret_cast<functor_type const*>(object);
        }
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1
      , typename IArchive, typename OArchive
    >
    struct function_base<
        R(A0 , A1)
      , IArchive, OArchive
    >
    {
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1)
              , IArchive, OArchive
            > vtable_ptr_type;
        
        function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * = 0
        ) : vptr(get_empty_table_ptr())
          , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename util::decay<Functor>::type
                    functor_type;
                vptr = get_table_ptr<functor_type>();
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
          : vptr(get_empty_table_ptr())
          , object(0)
        {
            assign(other);
        }
        function_base(function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        function_base &assign(function_base const & other)
        {
            if(&other != this)
            {
                if(vptr == other.vptr && !empty())
                {
                    vptr->copy(&other.object, &object);
                }
                else
                {
                    reset();
                    if(!other.empty())
                    {
                        other.vptr->clone(&other.object, &object);
                        vptr = other.vptr;
                    }
                }
            }
            return *this;
        }
        template <typename Functor>
        function_base & assign(Functor && f)
        {
            if (this == &f)
                return *this;
            typedef
                typename util::decay<Functor>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if(vptr == f_vptr && !empty())
            {
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    vptr->destruct(&object);
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(std::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(std::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        function_base & operator=(function_base const & t)
        {
            return assign(t);
        }
        function_base & operator=(function_base && t) BOOST_NOEXCEPT
        {
            if(this != &t)
            {
                reset();
                vptr = t.vptr;
                object = t.object;
                t.vptr = get_empty_table_ptr();
                t.object = 0;
            }
            return *this;
        }
        function_base &swap(function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const BOOST_NOEXCEPT
        {
            return empty();
        }
        void reset() BOOST_NOEXCEPT
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_empty_table<
                        R(A0 , A1)
                    >::template get<IArchive, OArchive>();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1)
                    >::template get<IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1) const
        {
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ));
        }
        std::type_info const& target_type() const BOOST_NOEXCEPT
        {
            return vptr->get_type();
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type *>(&object);
            return reinterpret_cast<functor_type *>(object);
        }
        template <typename T>
        T const* target() const BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type const*>(&object);
            return reinterpret_cast<functor_type const*>(object);
        }
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2
      , typename IArchive, typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2)
      , IArchive, OArchive
    >
    {
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2)
              , IArchive, OArchive
            > vtable_ptr_type;
        
        function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * = 0
        ) : vptr(get_empty_table_ptr())
          , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename util::decay<Functor>::type
                    functor_type;
                vptr = get_table_ptr<functor_type>();
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
          : vptr(get_empty_table_ptr())
          , object(0)
        {
            assign(other);
        }
        function_base(function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        function_base &assign(function_base const & other)
        {
            if(&other != this)
            {
                if(vptr == other.vptr && !empty())
                {
                    vptr->copy(&other.object, &object);
                }
                else
                {
                    reset();
                    if(!other.empty())
                    {
                        other.vptr->clone(&other.object, &object);
                        vptr = other.vptr;
                    }
                }
            }
            return *this;
        }
        template <typename Functor>
        function_base & assign(Functor && f)
        {
            if (this == &f)
                return *this;
            typedef
                typename util::decay<Functor>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if(vptr == f_vptr && !empty())
            {
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    vptr->destruct(&object);
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(std::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(std::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        function_base & operator=(function_base const & t)
        {
            return assign(t);
        }
        function_base & operator=(function_base && t) BOOST_NOEXCEPT
        {
            if(this != &t)
            {
                reset();
                vptr = t.vptr;
                object = t.object;
                t.vptr = get_empty_table_ptr();
                t.object = 0;
            }
            return *this;
        }
        function_base &swap(function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const BOOST_NOEXCEPT
        {
            return empty();
        }
        void reset() BOOST_NOEXCEPT
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_empty_table<
                        R(A0 , A1 , A2)
                    >::template get<IArchive, OArchive>();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2)
                    >::template get<IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2) const
        {
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ));
        }
        std::type_info const& target_type() const BOOST_NOEXCEPT
        {
            return vptr->get_type();
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type *>(&object);
            return reinterpret_cast<functor_type *>(object);
        }
        template <typename T>
        T const* target() const BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type const*>(&object);
            return reinterpret_cast<functor_type const*>(object);
        }
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3
      , typename IArchive, typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3)
      , IArchive, OArchive
    >
    {
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3)
              , IArchive, OArchive
            > vtable_ptr_type;
        
        function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * = 0
        ) : vptr(get_empty_table_ptr())
          , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename util::decay<Functor>::type
                    functor_type;
                vptr = get_table_ptr<functor_type>();
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
          : vptr(get_empty_table_ptr())
          , object(0)
        {
            assign(other);
        }
        function_base(function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        function_base &assign(function_base const & other)
        {
            if(&other != this)
            {
                if(vptr == other.vptr && !empty())
                {
                    vptr->copy(&other.object, &object);
                }
                else
                {
                    reset();
                    if(!other.empty())
                    {
                        other.vptr->clone(&other.object, &object);
                        vptr = other.vptr;
                    }
                }
            }
            return *this;
        }
        template <typename Functor>
        function_base & assign(Functor && f)
        {
            if (this == &f)
                return *this;
            typedef
                typename util::decay<Functor>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if(vptr == f_vptr && !empty())
            {
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    vptr->destruct(&object);
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(std::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(std::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        function_base & operator=(function_base const & t)
        {
            return assign(t);
        }
        function_base & operator=(function_base && t) BOOST_NOEXCEPT
        {
            if(this != &t)
            {
                reset();
                vptr = t.vptr;
                object = t.object;
                t.vptr = get_empty_table_ptr();
                t.object = 0;
            }
            return *this;
        }
        function_base &swap(function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const BOOST_NOEXCEPT
        {
            return empty();
        }
        void reset() BOOST_NOEXCEPT
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_empty_table<
                        R(A0 , A1 , A2 , A3)
                    >::template get<IArchive, OArchive>();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3)
                    >::template get<IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3) const
        {
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ));
        }
        std::type_info const& target_type() const BOOST_NOEXCEPT
        {
            return vptr->get_type();
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type *>(&object);
            return reinterpret_cast<functor_type *>(object);
        }
        template <typename T>
        T const* target() const BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type const*>(&object);
            return reinterpret_cast<functor_type const*>(object);
        }
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename IArchive, typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4)
      , IArchive, OArchive
    >
    {
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4)
              , IArchive, OArchive
            > vtable_ptr_type;
        
        function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * = 0
        ) : vptr(get_empty_table_ptr())
          , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename util::decay<Functor>::type
                    functor_type;
                vptr = get_table_ptr<functor_type>();
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
          : vptr(get_empty_table_ptr())
          , object(0)
        {
            assign(other);
        }
        function_base(function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        function_base &assign(function_base const & other)
        {
            if(&other != this)
            {
                if(vptr == other.vptr && !empty())
                {
                    vptr->copy(&other.object, &object);
                }
                else
                {
                    reset();
                    if(!other.empty())
                    {
                        other.vptr->clone(&other.object, &object);
                        vptr = other.vptr;
                    }
                }
            }
            return *this;
        }
        template <typename Functor>
        function_base & assign(Functor && f)
        {
            if (this == &f)
                return *this;
            typedef
                typename util::decay<Functor>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if(vptr == f_vptr && !empty())
            {
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    vptr->destruct(&object);
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(std::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(std::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        function_base & operator=(function_base const & t)
        {
            return assign(t);
        }
        function_base & operator=(function_base && t) BOOST_NOEXCEPT
        {
            if(this != &t)
            {
                reset();
                vptr = t.vptr;
                object = t.object;
                t.vptr = get_empty_table_ptr();
                t.object = 0;
            }
            return *this;
        }
        function_base &swap(function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const BOOST_NOEXCEPT
        {
            return empty();
        }
        void reset() BOOST_NOEXCEPT
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_empty_table<
                        R(A0 , A1 , A2 , A3 , A4)
                    >::template get<IArchive, OArchive>();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4)
                    >::template get<IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4) const
        {
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ));
        }
        std::type_info const& target_type() const BOOST_NOEXCEPT
        {
            return vptr->get_type();
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type *>(&object);
            return reinterpret_cast<functor_type *>(object);
        }
        template <typename T>
        T const* target() const BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type const*>(&object);
            return reinterpret_cast<functor_type const*>(object);
        }
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename IArchive, typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5)
      , IArchive, OArchive
    >
    {
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5)
              , IArchive, OArchive
            > vtable_ptr_type;
        
        function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * = 0
        ) : vptr(get_empty_table_ptr())
          , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename util::decay<Functor>::type
                    functor_type;
                vptr = get_table_ptr<functor_type>();
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
          : vptr(get_empty_table_ptr())
          , object(0)
        {
            assign(other);
        }
        function_base(function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        function_base &assign(function_base const & other)
        {
            if(&other != this)
            {
                if(vptr == other.vptr && !empty())
                {
                    vptr->copy(&other.object, &object);
                }
                else
                {
                    reset();
                    if(!other.empty())
                    {
                        other.vptr->clone(&other.object, &object);
                        vptr = other.vptr;
                    }
                }
            }
            return *this;
        }
        template <typename Functor>
        function_base & assign(Functor && f)
        {
            if (this == &f)
                return *this;
            typedef
                typename util::decay<Functor>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if(vptr == f_vptr && !empty())
            {
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    vptr->destruct(&object);
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(std::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(std::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        function_base & operator=(function_base const & t)
        {
            return assign(t);
        }
        function_base & operator=(function_base && t) BOOST_NOEXCEPT
        {
            if(this != &t)
            {
                reset();
                vptr = t.vptr;
                object = t.object;
                t.vptr = get_empty_table_ptr();
                t.object = 0;
            }
            return *this;
        }
        function_base &swap(function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const BOOST_NOEXCEPT
        {
            return empty();
        }
        void reset() BOOST_NOEXCEPT
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_empty_table<
                        R(A0 , A1 , A2 , A3 , A4 , A5)
                    >::template get<IArchive, OArchive>();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5)
                    >::template get<IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5) const
        {
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ));
        }
        std::type_info const& target_type() const BOOST_NOEXCEPT
        {
            return vptr->get_type();
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type *>(&object);
            return reinterpret_cast<functor_type *>(object);
        }
        template <typename T>
        T const* target() const BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type const*>(&object);
            return reinterpret_cast<functor_type const*>(object);
        }
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename IArchive, typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , IArchive, OArchive
    >
    {
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
              , IArchive, OArchive
            > vtable_ptr_type;
        
        function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * = 0
        ) : vptr(get_empty_table_ptr())
          , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename util::decay<Functor>::type
                    functor_type;
                vptr = get_table_ptr<functor_type>();
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
          : vptr(get_empty_table_ptr())
          , object(0)
        {
            assign(other);
        }
        function_base(function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        function_base &assign(function_base const & other)
        {
            if(&other != this)
            {
                if(vptr == other.vptr && !empty())
                {
                    vptr->copy(&other.object, &object);
                }
                else
                {
                    reset();
                    if(!other.empty())
                    {
                        other.vptr->clone(&other.object, &object);
                        vptr = other.vptr;
                    }
                }
            }
            return *this;
        }
        template <typename Functor>
        function_base & assign(Functor && f)
        {
            if (this == &f)
                return *this;
            typedef
                typename util::decay<Functor>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if(vptr == f_vptr && !empty())
            {
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    vptr->destruct(&object);
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(std::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(std::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        function_base & operator=(function_base const & t)
        {
            return assign(t);
        }
        function_base & operator=(function_base && t) BOOST_NOEXCEPT
        {
            if(this != &t)
            {
                reset();
                vptr = t.vptr;
                object = t.object;
                t.vptr = get_empty_table_ptr();
                t.object = 0;
            }
            return *this;
        }
        function_base &swap(function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const BOOST_NOEXCEPT
        {
            return empty();
        }
        void reset() BOOST_NOEXCEPT
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_empty_table<
                        R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
                    >::template get<IArchive, OArchive>();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
                    >::template get<IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6) const
        {
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ));
        }
        std::type_info const& target_type() const BOOST_NOEXCEPT
        {
            return vptr->get_type();
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type *>(&object);
            return reinterpret_cast<functor_type *>(object);
        }
        template <typename T>
        T const* target() const BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type const*>(&object);
            return reinterpret_cast<functor_type const*>(object);
        }
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename IArchive, typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
      , IArchive, OArchive
    >
    {
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
              , IArchive, OArchive
            > vtable_ptr_type;
        
        function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * = 0
        ) : vptr(get_empty_table_ptr())
          , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename util::decay<Functor>::type
                    functor_type;
                vptr = get_table_ptr<functor_type>();
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
          : vptr(get_empty_table_ptr())
          , object(0)
        {
            assign(other);
        }
        function_base(function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        function_base &assign(function_base const & other)
        {
            if(&other != this)
            {
                if(vptr == other.vptr && !empty())
                {
                    vptr->copy(&other.object, &object);
                }
                else
                {
                    reset();
                    if(!other.empty())
                    {
                        other.vptr->clone(&other.object, &object);
                        vptr = other.vptr;
                    }
                }
            }
            return *this;
        }
        template <typename Functor>
        function_base & assign(Functor && f)
        {
            if (this == &f)
                return *this;
            typedef
                typename util::decay<Functor>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if(vptr == f_vptr && !empty())
            {
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    vptr->destruct(&object);
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(std::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(std::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        function_base & operator=(function_base const & t)
        {
            return assign(t);
        }
        function_base & operator=(function_base && t) BOOST_NOEXCEPT
        {
            if(this != &t)
            {
                reset();
                vptr = t.vptr;
                object = t.object;
                t.vptr = get_empty_table_ptr();
                t.object = 0;
            }
            return *this;
        }
        function_base &swap(function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const BOOST_NOEXCEPT
        {
            return empty();
        }
        void reset() BOOST_NOEXCEPT
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_empty_table<
                        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
                    >::template get<IArchive, OArchive>();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
                    >::template get<IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7) const
        {
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ));
        }
        std::type_info const& target_type() const BOOST_NOEXCEPT
        {
            return vptr->get_type();
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type *>(&object);
            return reinterpret_cast<functor_type *>(object);
        }
        template <typename T>
        T const* target() const BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type const*>(&object);
            return reinterpret_cast<functor_type const*>(object);
        }
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename IArchive, typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
      , IArchive, OArchive
    >
    {
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
              , IArchive, OArchive
            > vtable_ptr_type;
        
        function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * = 0
        ) : vptr(get_empty_table_ptr())
          , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename util::decay<Functor>::type
                    functor_type;
                vptr = get_table_ptr<functor_type>();
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
          : vptr(get_empty_table_ptr())
          , object(0)
        {
            assign(other);
        }
        function_base(function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        function_base &assign(function_base const & other)
        {
            if(&other != this)
            {
                if(vptr == other.vptr && !empty())
                {
                    vptr->copy(&other.object, &object);
                }
                else
                {
                    reset();
                    if(!other.empty())
                    {
                        other.vptr->clone(&other.object, &object);
                        vptr = other.vptr;
                    }
                }
            }
            return *this;
        }
        template <typename Functor>
        function_base & assign(Functor && f)
        {
            if (this == &f)
                return *this;
            typedef
                typename util::decay<Functor>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if(vptr == f_vptr && !empty())
            {
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    vptr->destruct(&object);
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(std::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(std::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        function_base & operator=(function_base const & t)
        {
            return assign(t);
        }
        function_base & operator=(function_base && t) BOOST_NOEXCEPT
        {
            if(this != &t)
            {
                reset();
                vptr = t.vptr;
                object = t.object;
                t.vptr = get_empty_table_ptr();
                t.object = 0;
            }
            return *this;
        }
        function_base &swap(function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const BOOST_NOEXCEPT
        {
            return empty();
        }
        void reset() BOOST_NOEXCEPT
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_empty_table<
                        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
                    >::template get<IArchive, OArchive>();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
                    >::template get<IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8) const
        {
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ));
        }
        std::type_info const& target_type() const BOOST_NOEXCEPT
        {
            return vptr->get_type();
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type *>(&object);
            return reinterpret_cast<functor_type *>(object);
        }
        template <typename T>
        T const* target() const BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type const*>(&object);
            return reinterpret_cast<functor_type const*>(object);
        }
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename IArchive, typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
      , IArchive, OArchive
    >
    {
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
              , IArchive, OArchive
            > vtable_ptr_type;
        
        function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * = 0
        ) : vptr(get_empty_table_ptr())
          , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename util::decay<Functor>::type
                    functor_type;
                vptr = get_table_ptr<functor_type>();
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
          : vptr(get_empty_table_ptr())
          , object(0)
        {
            assign(other);
        }
        function_base(function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        function_base &assign(function_base const & other)
        {
            if(&other != this)
            {
                if(vptr == other.vptr && !empty())
                {
                    vptr->copy(&other.object, &object);
                }
                else
                {
                    reset();
                    if(!other.empty())
                    {
                        other.vptr->clone(&other.object, &object);
                        vptr = other.vptr;
                    }
                }
            }
            return *this;
        }
        template <typename Functor>
        function_base & assign(Functor && f)
        {
            if (this == &f)
                return *this;
            typedef
                typename util::decay<Functor>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if(vptr == f_vptr && !empty())
            {
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    vptr->destruct(&object);
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(std::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(std::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        function_base & operator=(function_base const & t)
        {
            return assign(t);
        }
        function_base & operator=(function_base && t) BOOST_NOEXCEPT
        {
            if(this != &t)
            {
                reset();
                vptr = t.vptr;
                object = t.object;
                t.vptr = get_empty_table_ptr();
                t.object = 0;
            }
            return *this;
        }
        function_base &swap(function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const BOOST_NOEXCEPT
        {
            return empty();
        }
        void reset() BOOST_NOEXCEPT
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_empty_table<
                        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
                    >::template get<IArchive, OArchive>();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
                    >::template get<IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9) const
        {
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ));
        }
        std::type_info const& target_type() const BOOST_NOEXCEPT
        {
            return vptr->get_type();
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type *>(&object);
            return reinterpret_cast<functor_type *>(object);
        }
        template <typename T>
        T const* target() const BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type const*>(&object);
            return reinterpret_cast<functor_type const*>(object);
        }
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename IArchive, typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
      , IArchive, OArchive
    >
    {
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
              , IArchive, OArchive
            > vtable_ptr_type;
        
        function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * = 0
        ) : vptr(get_empty_table_ptr())
          , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename util::decay<Functor>::type
                    functor_type;
                vptr = get_table_ptr<functor_type>();
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
          : vptr(get_empty_table_ptr())
          , object(0)
        {
            assign(other);
        }
        function_base(function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        function_base &assign(function_base const & other)
        {
            if(&other != this)
            {
                if(vptr == other.vptr && !empty())
                {
                    vptr->copy(&other.object, &object);
                }
                else
                {
                    reset();
                    if(!other.empty())
                    {
                        other.vptr->clone(&other.object, &object);
                        vptr = other.vptr;
                    }
                }
            }
            return *this;
        }
        template <typename Functor>
        function_base & assign(Functor && f)
        {
            if (this == &f)
                return *this;
            typedef
                typename util::decay<Functor>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if(vptr == f_vptr && !empty())
            {
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    vptr->destruct(&object);
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(std::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(std::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        function_base & operator=(function_base const & t)
        {
            return assign(t);
        }
        function_base & operator=(function_base && t) BOOST_NOEXCEPT
        {
            if(this != &t)
            {
                reset();
                vptr = t.vptr;
                object = t.object;
                t.vptr = get_empty_table_ptr();
                t.object = 0;
            }
            return *this;
        }
        function_base &swap(function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const BOOST_NOEXCEPT
        {
            return empty();
        }
        void reset() BOOST_NOEXCEPT
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_empty_table<
                        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
                    >::template get<IArchive, OArchive>();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
                    >::template get<IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10) const
        {
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ));
        }
        std::type_info const& target_type() const BOOST_NOEXCEPT
        {
            return vptr->get_type();
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type *>(&object);
            return reinterpret_cast<functor_type *>(object);
        }
        template <typename T>
        T const* target() const BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type const*>(&object);
            return reinterpret_cast<functor_type const*>(object);
        }
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename IArchive, typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
      , IArchive, OArchive
    >
    {
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
              , IArchive, OArchive
            > vtable_ptr_type;
        
        function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * = 0
        ) : vptr(get_empty_table_ptr())
          , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename util::decay<Functor>::type
                    functor_type;
                vptr = get_table_ptr<functor_type>();
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
          : vptr(get_empty_table_ptr())
          , object(0)
        {
            assign(other);
        }
        function_base(function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        function_base &assign(function_base const & other)
        {
            if(&other != this)
            {
                if(vptr == other.vptr && !empty())
                {
                    vptr->copy(&other.object, &object);
                }
                else
                {
                    reset();
                    if(!other.empty())
                    {
                        other.vptr->clone(&other.object, &object);
                        vptr = other.vptr;
                    }
                }
            }
            return *this;
        }
        template <typename Functor>
        function_base & assign(Functor && f)
        {
            if (this == &f)
                return *this;
            typedef
                typename util::decay<Functor>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if(vptr == f_vptr && !empty())
            {
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    vptr->destruct(&object);
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(std::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(std::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        function_base & operator=(function_base const & t)
        {
            return assign(t);
        }
        function_base & operator=(function_base && t) BOOST_NOEXCEPT
        {
            if(this != &t)
            {
                reset();
                vptr = t.vptr;
                object = t.object;
                t.vptr = get_empty_table_ptr();
                t.object = 0;
            }
            return *this;
        }
        function_base &swap(function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const BOOST_NOEXCEPT
        {
            return empty();
        }
        void reset() BOOST_NOEXCEPT
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_empty_table<
                        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
                    >::template get<IArchive, OArchive>();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
                    >::template get<IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11) const
        {
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ));
        }
        std::type_info const& target_type() const BOOST_NOEXCEPT
        {
            return vptr->get_type();
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type *>(&object);
            return reinterpret_cast<functor_type *>(object);
        }
        template <typename T>
        T const* target() const BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type const*>(&object);
            return reinterpret_cast<functor_type const*>(object);
        }
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}}
namespace hpx { namespace util { namespace detail
{
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
      , typename IArchive, typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
      , IArchive, OArchive
    >
    {
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
              , IArchive, OArchive
            > vtable_ptr_type;
        
        function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * = 0
        ) : vptr(get_empty_table_ptr())
          , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename util::decay<Functor>::type
                    functor_type;
                vptr = get_table_ptr<functor_type>();
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
          : vptr(get_empty_table_ptr())
          , object(0)
        {
            assign(other);
        }
        function_base(function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        function_base &assign(function_base const & other)
        {
            if(&other != this)
            {
                if(vptr == other.vptr && !empty())
                {
                    vptr->copy(&other.object, &object);
                }
                else
                {
                    reset();
                    if(!other.empty())
                    {
                        other.vptr->clone(&other.object, &object);
                        vptr = other.vptr;
                    }
                }
            }
            return *this;
        }
        template <typename Functor>
        function_base & assign(Functor && f)
        {
            if (this == &f)
                return *this;
            typedef
                typename util::decay<Functor>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if(vptr == f_vptr && !empty())
            {
                if (sizeof(functor_type) <= sizeof(void *)) 
                {
                    vptr->destruct(&object);
                    new (&object) functor_type(std::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(std::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(std::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(std::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(std::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        function_base & operator=(function_base const & t)
        {
            return assign(t);
        }
        function_base & operator=(function_base && t) BOOST_NOEXCEPT
        {
            if(this != &t)
            {
                reset();
                vptr = t.vptr;
                object = t.object;
                t.vptr = get_empty_table_ptr();
                t.object = 0;
            }
            return *this;
        }
        function_base &swap(function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const BOOST_NOEXCEPT
        {
            return empty();
        }
        void reset() BOOST_NOEXCEPT
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_empty_table<
                        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
                    >::template get<IArchive, OArchive>();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
                    >::template get<IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12) const
        {
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ));
        }
        std::type_info const& target_type() const BOOST_NOEXCEPT
        {
            return vptr->get_type();
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type *>(&object);
            return reinterpret_cast<functor_type *>(object);
        }
        template <typename T>
        T const* target() const BOOST_NOEXCEPT
        {
            typedef
                typename util::decay<T>::type
                functor_type;
            vtable_ptr_type* f_vptr = get_table_ptr<functor_type>();
            if (vptr != f_vptr || empty())
                return 0;
            if (sizeof(functor_type) <= sizeof(void *)) 
                return reinterpret_cast<functor_type const*>(&object);
            return reinterpret_cast<functor_type const*>(object);
        }
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}}
