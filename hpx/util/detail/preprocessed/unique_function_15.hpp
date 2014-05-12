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
    struct unique_function_base<
        R()
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);
    public:
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
        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
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
        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        template <typename Functor>
        unique_function_base & assign(Functor && f)
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
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
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
        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
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
                    >::template get<true, IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()()
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                 );
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
    struct unique_function_base<
        R(A0)
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);
    public:
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
        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
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
        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        template <typename Functor>
        unique_function_base & assign(Functor && f)
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
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
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
        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
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
                    >::template get<true, IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0)
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                , std::forward<A0>( a0 ));
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
    struct unique_function_base<
        R(A0 , A1)
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);
    public:
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
        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
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
        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        template <typename Functor>
        unique_function_base & assign(Functor && f)
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
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
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
        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
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
                    >::template get<true, IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1)
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ));
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
    struct unique_function_base<
        R(A0 , A1 , A2)
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);
    public:
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
        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
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
        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        template <typename Functor>
        unique_function_base & assign(Functor && f)
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
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
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
        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
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
                    >::template get<true, IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2)
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ));
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
    struct unique_function_base<
        R(A0 , A1 , A2 , A3)
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);
    public:
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
        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
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
        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        template <typename Functor>
        unique_function_base & assign(Functor && f)
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
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
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
        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
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
                    >::template get<true, IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3)
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ));
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
    struct unique_function_base<
        R(A0 , A1 , A2 , A3 , A4)
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);
    public:
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
        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
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
        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        template <typename Functor>
        unique_function_base & assign(Functor && f)
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
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
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
        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
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
                    >::template get<true, IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4)
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ));
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
    struct unique_function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5)
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);
    public:
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
        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
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
        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        template <typename Functor>
        unique_function_base & assign(Functor && f)
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
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
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
        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
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
                    >::template get<true, IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5)
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ));
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
    struct unique_function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);
    public:
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
        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
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
        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        template <typename Functor>
        unique_function_base & assign(Functor && f)
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
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
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
        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
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
                    >::template get<true, IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6)
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ));
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
    struct unique_function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);
    public:
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
        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
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
        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        template <typename Functor>
        unique_function_base & assign(Functor && f)
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
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
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
        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
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
                    >::template get<true, IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7)
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ));
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
    struct unique_function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);
    public:
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
        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
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
        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        template <typename Functor>
        unique_function_base & assign(Functor && f)
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
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
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
        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
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
                    >::template get<true, IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8)
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ));
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
    struct unique_function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);
    public:
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
        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
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
        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        template <typename Functor>
        unique_function_base & assign(Functor && f)
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
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
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
        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
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
                    >::template get<true, IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9)
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ));
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
    struct unique_function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);
    public:
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
        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
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
        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        template <typename Functor>
        unique_function_base & assign(Functor && f)
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
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
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
        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
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
                    >::template get<true, IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10)
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ));
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
    struct unique_function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);
    public:
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
        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
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
        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        template <typename Functor>
        unique_function_base & assign(Functor && f)
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
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
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
        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
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
                    >::template get<true, IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11)
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ));
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
    struct unique_function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);
    public:
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
        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
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
        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        template <typename Functor>
        unique_function_base & assign(Functor && f)
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
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
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
        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
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
                    >::template get<true, IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12)
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ));
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
      , typename IArchive, typename OArchive
    >
    struct unique_function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);
    public:
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
              , IArchive, OArchive
            > vtable_ptr_type;
        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
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
        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        template <typename Functor>
        unique_function_base & assign(Functor && f)
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
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
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
        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
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
                        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
                    >::template get<IArchive, OArchive>();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
                    >::template get<true, IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13)
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ));
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
      , typename IArchive, typename OArchive
    >
    struct unique_function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);
    public:
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
              , IArchive, OArchive
            > vtable_ptr_type;
        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
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
        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        template <typename Functor>
        unique_function_base & assign(Functor && f)
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
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
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
        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
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
                        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
                    >::template get<IArchive, OArchive>();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
                    >::template get<true, IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14)
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ));
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
      , typename IArchive, typename OArchive
    >
    struct unique_function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);
    public:
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
              , IArchive, OArchive
            > vtable_ptr_type;
        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
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
        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        template <typename Functor>
        unique_function_base & assign(Functor && f)
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
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
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
        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
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
                        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
                    >::template get<IArchive, OArchive>();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
                    >::template get<true, IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15)
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ));
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
      , typename IArchive, typename OArchive
    >
    struct unique_function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);
    public:
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
              , IArchive, OArchive
            > vtable_ptr_type;
        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
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
        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        template <typename Functor>
        unique_function_base & assign(Functor && f)
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
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
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
        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
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
                        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
                    >::template get<IArchive, OArchive>();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
                    >::template get<true, IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16)
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ));
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17
      , typename IArchive, typename OArchive
    >
    struct unique_function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
      , IArchive, OArchive
    >
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_base);
    public:
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive, OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
              , IArchive, OArchive
            > vtable_ptr_type;
        unique_function_base() BOOST_NOEXCEPT
          : vptr(get_empty_table_ptr())
          , object(0)
        {}
        ~unique_function_base()
        {
            if(object)
            {
                vptr->static_delete(&object);
            }
        }
        template <typename Functor>
        explicit unique_function_base(
            Functor && f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    unique_function_base
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
        unique_function_base(unique_function_base && other) BOOST_NOEXCEPT
          : vptr(other.vptr)
          , object(other.object)
        {
            other.vptr = get_empty_table_ptr();
            other.object = 0;
        }
        template <typename Functor>
        unique_function_base & assign(Functor && f)
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
        unique_function_base & operator=(T && t)
        {
            return assign(std::forward<T>(t));
        }
        unique_function_base & operator=(unique_function_base && t) BOOST_NOEXCEPT
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
        unique_function_base &swap(unique_function_base& f) BOOST_NOEXCEPT
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);
            return *this;
        }
        bool empty() const BOOST_NOEXCEPT
        {
            return object == 0 && vptr->empty();
        }
        operator typename util::safe_bool<unique_function_base>::result_type() const BOOST_NOEXCEPT
        {
            return util::safe_bool<unique_function_base>()(!empty());
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
                        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
                    >::template get<IArchive, OArchive>();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr() BOOST_NOEXCEPT
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
                    >::template get<true, IArchive, OArchive>();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16 , A17 a17)
        {
            invalidate_function<unique_function_base> on_exit(*this);
            return vptr->invoke(&object
                , std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 ));
        }
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}}
