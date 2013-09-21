// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace util {
    template <
        typename R
      
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R()
      , IArchive
      , OArchive
    >
    {
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
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive
              , OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R()
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * dummy = 0
        )
            : vptr(get_empty_table_ptr())
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
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
            : vptr(get_empty_table_ptr())
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
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
        function_base & assign(BOOST_FWD_REF(Functor) f)
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
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(boost::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(boost::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(BOOST_FWD_REF(T) t)
        {
            return assign(boost::forward<T>(t));
        }
        function_base & operator=(BOOST_COPY_ASSIGN_REF(function_base) t)
        {
            return assign(t);
        }
        function_base & operator=(BOOST_RV_REF(function_base) t)
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
        function_base &swap(function_base& f)
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
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr()
        {
            return detail::get_empty_table<
                        R()
                    >::template get<
                        IArchive
                      , OArchive
                    >();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr()
        {
            return detail::get_table<
                        Functor
                      , R()
                    >::template get<
                        IArchive
                      , OArchive
                    >();
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
    private:
        BOOST_COPYABLE_AND_MOVABLE(function_base);
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}
namespace hpx { namespace util {
    template <
        typename R
      , typename A0
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0)
      , IArchive
      , OArchive
    >
    {
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
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive
              , OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * dummy = 0
        )
            : vptr(get_empty_table_ptr())
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
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
            : vptr(get_empty_table_ptr())
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
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
        function_base & assign(BOOST_FWD_REF(Functor) f)
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
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(boost::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(boost::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(BOOST_FWD_REF(T) t)
        {
            return assign(boost::forward<T>(t));
        }
        function_base & operator=(BOOST_COPY_ASSIGN_REF(function_base) t)
        {
            return assign(t);
        }
        function_base & operator=(BOOST_RV_REF(function_base) t)
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
        function_base &swap(function_base& f)
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
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr()
        {
            return detail::get_empty_table<
                        R(A0)
                    >::template get<
                        IArchive
                      , OArchive
                    >();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr()
        {
            return detail::get_table<
                        Functor
                      , R(A0)
                    >::template get<
                        IArchive
                      , OArchive
                    >();
        }
        BOOST_FORCEINLINE R operator()(A0 a0) const
        {
            return vptr->invoke(&object
                , boost::forward<A0>( a0 ));
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
    private:
        BOOST_COPYABLE_AND_MOVABLE(function_base);
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}
namespace hpx { namespace util {
    template <
        typename R
      , typename A0 , typename A1
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1)
      , IArchive
      , OArchive
    >
    {
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
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive
              , OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * dummy = 0
        )
            : vptr(get_empty_table_ptr())
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
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
            : vptr(get_empty_table_ptr())
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
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
        function_base & assign(BOOST_FWD_REF(Functor) f)
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
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(boost::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(boost::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(BOOST_FWD_REF(T) t)
        {
            return assign(boost::forward<T>(t));
        }
        function_base & operator=(BOOST_COPY_ASSIGN_REF(function_base) t)
        {
            return assign(t);
        }
        function_base & operator=(BOOST_RV_REF(function_base) t)
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
        function_base &swap(function_base& f)
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
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr()
        {
            return detail::get_empty_table<
                        R(A0 , A1)
                    >::template get<
                        IArchive
                      , OArchive
                    >();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr()
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1)
                    >::template get<
                        IArchive
                      , OArchive
                    >();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1) const
        {
            return vptr->invoke(&object
                , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ));
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
    private:
        BOOST_COPYABLE_AND_MOVABLE(function_base);
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}
namespace hpx { namespace util {
    template <
        typename R
      , typename A0 , typename A1 , typename A2
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2)
      , IArchive
      , OArchive
    >
    {
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
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive
              , OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * dummy = 0
        )
            : vptr(get_empty_table_ptr())
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
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
            : vptr(get_empty_table_ptr())
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
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
        function_base & assign(BOOST_FWD_REF(Functor) f)
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
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(boost::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(boost::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(BOOST_FWD_REF(T) t)
        {
            return assign(boost::forward<T>(t));
        }
        function_base & operator=(BOOST_COPY_ASSIGN_REF(function_base) t)
        {
            return assign(t);
        }
        function_base & operator=(BOOST_RV_REF(function_base) t)
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
        function_base &swap(function_base& f)
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
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr()
        {
            return detail::get_empty_table<
                        R(A0 , A1 , A2)
                    >::template get<
                        IArchive
                      , OArchive
                    >();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr()
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2)
                    >::template get<
                        IArchive
                      , OArchive
                    >();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2) const
        {
            return vptr->invoke(&object
                , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ));
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
    private:
        BOOST_COPYABLE_AND_MOVABLE(function_base);
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}
namespace hpx { namespace util {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3)
      , IArchive
      , OArchive
    >
    {
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
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive
              , OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * dummy = 0
        )
            : vptr(get_empty_table_ptr())
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
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
            : vptr(get_empty_table_ptr())
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
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
        function_base & assign(BOOST_FWD_REF(Functor) f)
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
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(boost::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(boost::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(BOOST_FWD_REF(T) t)
        {
            return assign(boost::forward<T>(t));
        }
        function_base & operator=(BOOST_COPY_ASSIGN_REF(function_base) t)
        {
            return assign(t);
        }
        function_base & operator=(BOOST_RV_REF(function_base) t)
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
        function_base &swap(function_base& f)
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
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr()
        {
            return detail::get_empty_table<
                        R(A0 , A1 , A2 , A3)
                    >::template get<
                        IArchive
                      , OArchive
                    >();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr()
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3)
                    >::template get<
                        IArchive
                      , OArchive
                    >();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3) const
        {
            return vptr->invoke(&object
                , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ));
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
    private:
        BOOST_COPYABLE_AND_MOVABLE(function_base);
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}
namespace hpx { namespace util {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4)
      , IArchive
      , OArchive
    >
    {
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
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive
              , OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * dummy = 0
        )
            : vptr(get_empty_table_ptr())
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
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
            : vptr(get_empty_table_ptr())
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
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
        function_base & assign(BOOST_FWD_REF(Functor) f)
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
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(boost::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(boost::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(BOOST_FWD_REF(T) t)
        {
            return assign(boost::forward<T>(t));
        }
        function_base & operator=(BOOST_COPY_ASSIGN_REF(function_base) t)
        {
            return assign(t);
        }
        function_base & operator=(BOOST_RV_REF(function_base) t)
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
        function_base &swap(function_base& f)
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
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr()
        {
            return detail::get_empty_table<
                        R(A0 , A1 , A2 , A3 , A4)
                    >::template get<
                        IArchive
                      , OArchive
                    >();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr()
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4)
                    >::template get<
                        IArchive
                      , OArchive
                    >();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4) const
        {
            return vptr->invoke(&object
                , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ));
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
    private:
        BOOST_COPYABLE_AND_MOVABLE(function_base);
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}
namespace hpx { namespace util {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5)
      , IArchive
      , OArchive
    >
    {
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
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive
              , OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * dummy = 0
        )
            : vptr(get_empty_table_ptr())
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
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
            : vptr(get_empty_table_ptr())
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
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
        function_base & assign(BOOST_FWD_REF(Functor) f)
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
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(boost::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(boost::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(BOOST_FWD_REF(T) t)
        {
            return assign(boost::forward<T>(t));
        }
        function_base & operator=(BOOST_COPY_ASSIGN_REF(function_base) t)
        {
            return assign(t);
        }
        function_base & operator=(BOOST_RV_REF(function_base) t)
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
        function_base &swap(function_base& f)
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
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr()
        {
            return detail::get_empty_table<
                        R(A0 , A1 , A2 , A3 , A4 , A5)
                    >::template get<
                        IArchive
                      , OArchive
                    >();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr()
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5)
                    >::template get<
                        IArchive
                      , OArchive
                    >();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5) const
        {
            return vptr->invoke(&object
                , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ));
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
    private:
        BOOST_COPYABLE_AND_MOVABLE(function_base);
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}
namespace hpx { namespace util {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
      , IArchive
      , OArchive
    >
    {
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
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive
              , OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * dummy = 0
        )
            : vptr(get_empty_table_ptr())
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
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
            : vptr(get_empty_table_ptr())
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
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
        function_base & assign(BOOST_FWD_REF(Functor) f)
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
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(boost::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(boost::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(BOOST_FWD_REF(T) t)
        {
            return assign(boost::forward<T>(t));
        }
        function_base & operator=(BOOST_COPY_ASSIGN_REF(function_base) t)
        {
            return assign(t);
        }
        function_base & operator=(BOOST_RV_REF(function_base) t)
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
        function_base &swap(function_base& f)
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
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr()
        {
            return detail::get_empty_table<
                        R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
                    >::template get<
                        IArchive
                      , OArchive
                    >();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr()
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
                    >::template get<
                        IArchive
                      , OArchive
                    >();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6) const
        {
            return vptr->invoke(&object
                , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ));
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
    private:
        BOOST_COPYABLE_AND_MOVABLE(function_base);
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}
namespace hpx { namespace util {
    template <
        typename R
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
      , IArchive
      , OArchive
    >
    {
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
        typedef R result_type;
        typedef
            detail::vtable_ptr_virtbase<
                IArchive
              , OArchive
            > vtable_virtbase_type;
        typedef
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename util::decay<Functor>::type
                >::type
            >::type * dummy = 0
        )
            : vptr(get_empty_table_ptr())
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
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
        }
        function_base(function_base const & other)
            : vptr(get_empty_table_ptr())
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
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
        function_base & assign(BOOST_FWD_REF(Functor) f)
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
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else if (object)
                {
                    vptr->destruct(&object);
                    new (object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                reset();
                if (!detail::is_empty_function(f))
                {
                    if (sizeof(functor_type) <= sizeof(void *)) 
                    {
                        new (&object) functor_type(boost::forward<Functor>(f));
                    }
                    else
                    {
                        object = new functor_type(boost::forward<Functor>(f));
                    }
                    vptr = f_vptr;
                }
            }
            return *this;
        }
        template <typename T>
        function_base & operator=(BOOST_FWD_REF(T) t)
        {
            return assign(boost::forward<T>(t));
        }
        function_base & operator=(BOOST_COPY_ASSIGN_REF(function_base) t)
        {
            return assign(t);
        }
        function_base & operator=(BOOST_RV_REF(function_base) t)
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
        function_base &swap(function_base& f)
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
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = get_empty_table_ptr();
                object = 0;
            }
        }
        static vtable_ptr_type* get_empty_table_ptr()
        {
            return detail::get_empty_table<
                        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
                    >::template get<
                        IArchive
                      , OArchive
                    >();
        }
        template <typename Functor>
        static vtable_ptr_type* get_table_ptr()
        {
            return detail::get_table<
                        Functor
                      , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
                    >::template get<
                        IArchive
                      , OArchive
                    >();
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7) const
        {
            return vptr->invoke(&object
                , boost::forward<A0>( a0 ) , boost::forward<A1>( a1 ) , boost::forward<A2>( a2 ) , boost::forward<A3>( a3 ) , boost::forward<A4>( a4 ) , boost::forward<A5>( a5 ) , boost::forward<A6>( a6 ) , boost::forward<A7>( a7 ));
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
    private:
        BOOST_COPYABLE_AND_MOVABLE(function_base);
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}
