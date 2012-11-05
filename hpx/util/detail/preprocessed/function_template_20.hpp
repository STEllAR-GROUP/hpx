// Copyright (c) 2007-2012 Hartmut Kaiser
// Copyright (c)      2012 Thomas Heller
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
        function_base()
            : vptr(0)
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
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R()
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R()>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()() const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object );
        }
        BOOST_FORCEINLINE R operator()()
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object );
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
        function_base()
            : vptr(0)
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
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0);
        }
        BOOST_FORCEINLINE R operator()(A0 a0)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0);
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
        function_base()
            : vptr(0)
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
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1);
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
        function_base()
            : vptr(0)
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
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2);
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
        function_base()
            : vptr(0)
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
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3);
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
        function_base()
            : vptr(0)
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
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3 , A4)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3 , A4)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4);
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
        function_base()
            : vptr(0)
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
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3 , A4 , A5)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3 , A4 , A5)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5);
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
        function_base()
            : vptr(0)
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
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3 , A4 , A5 , A6)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6);
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
        function_base()
            : vptr(0)
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
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7);
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
      , IArchive
      , OArchive
    >
    {
        function_base()
            : vptr(0)
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
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8);
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
      , IArchive
      , OArchive
    >
    {
        function_base()
            : vptr(0)
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
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9);
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
      , IArchive
      , OArchive
    >
    {
        function_base()
            : vptr(0)
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
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10);
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
      , IArchive
      , OArchive
    >
    {
        function_base()
            : vptr(0)
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
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11);
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
      , IArchive
      , OArchive
    >
    {
        function_base()
            : vptr(0)
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
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12);
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
      , IArchive
      , OArchive
    >
    {
        function_base()
            : vptr(0)
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
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13);
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
      , IArchive
      , OArchive
    >
    {
        function_base()
            : vptr(0)
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
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14);
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
      , IArchive
      , OArchive
    >
    {
        function_base()
            : vptr(0)
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
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15);
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
      , IArchive
      , OArchive
    >
    {
        function_base()
            : vptr(0)
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
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16);
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
      , IArchive
      , OArchive
    >
    {
        function_base()
            : vptr(0)
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
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16 , A17 a17) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16 , A17 a17)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17);
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18)
      , IArchive
      , OArchive
    >
    {
        function_base()
            : vptr(0)
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
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16 , A17 a17 , A18 a18) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16 , A17 a17 , A18 a18)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18);
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19)
      , IArchive
      , OArchive
    >
    {
        function_base()
            : vptr(0)
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
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16 , A17 a17 , A18 a18 , A19 a19) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18 , a19);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16 , A17 a17 , A18 a18 , A19 a19)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18 , a19);
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20)
      , IArchive
      , OArchive
    >
    {
        function_base()
            : vptr(0)
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
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16 , A17 a17 , A18 a18 , A19 a19 , A20 a20) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18 , a19 , a20);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16 , A17 a17 , A18 a18 , A19 a19 , A20 a20)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18 , a19 , a20);
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20 , typename A21
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21)
      , IArchive
      , OArchive
    >
    {
        function_base()
            : vptr(0)
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
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16 , A17 a17 , A18 a18 , A19 a19 , A20 a20 , A21 a21) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18 , a19 , a20 , a21);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16 , A17 a17 , A18 a18 , A19 a19 , A20 a20 , A21 a21)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18 , a19 , a20 , a21);
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
      , typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17 , typename A18 , typename A19 , typename A20 , typename A21 , typename A22
      , typename IArchive
      , typename OArchive
    >
    struct function_base<
        R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21 , A22)
      , IArchive
      , OArchive
    >
    {
        function_base()
            : vptr(0)
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
            detail::vtable_ptr_base<
                R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21 , A22)
              , IArchive
              , OArchive
            > vtable_ptr_type;
        template <typename Functor>
        explicit function_base(
            BOOST_FWD_REF(Functor) f
          , typename ::boost::disable_if<
                typename boost::is_same<
                    function_base
                  , typename boost::remove_const<
                        typename hpx::util::detail::remove_reference<
                            Functor
                        >::type
                    >::type
                >::type
            >::type * dummy = 0
        )
            : vptr(0)
            , object(0)
        {
            if (!detail::is_empty_function(f))
            {
                typedef
                    typename boost::remove_const<
                        typename boost::decay<
                            typename hpx::util::detail::remove_reference<Functor>::type
                        >::type
                    >::type
                    functor_type;
                vptr = detail::get_table<
                            functor_type
                          , R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21 , A22)
                        >::template get<
                            IArchive
                          , OArchive
                        >();
                const bool is_small = sizeof(functor_type) <= sizeof(void *);
                if(is_small)
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
            : vptr(0)
            , object(0)
        {
            assign(other);
        }
        function_base(BOOST_RV_REF(function_base) other)
            : vptr(other.vptr)
            , object(other.object)
        {
            other.vptr = 0;
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
            typedef
                typename boost::remove_const<
                    typename boost::decay<
                        typename hpx::util::detail::remove_reference<Functor>::type
                    >::type
                >::type
                functor_type;
            const bool is_small = sizeof(functor_type) <= sizeof(void *);
            vtable_ptr_type * f_vptr
                = detail::get_table<functor_type, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17 , A18 , A19 , A20 , A21 , A22)>::template get<
                    IArchive
                  , OArchive
                >();
            if(vptr == f_vptr && !empty())
            {
                vptr->destruct(&object);
                if(is_small)
                {
                    new (&object) functor_type(boost::forward<Functor>(f));
                }
                else
                {
                    object = new functor_type(boost::forward<Functor>(f));
                }
            }
            else
            {
                if(!empty())
                {
                    vptr->destruct(&object);
                    vptr = 0;
                    object = 0;
                }
                if (!detail::is_empty_function(f))
                {
                    if(is_small)
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
                t.vptr = 0;
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
        bool empty() const
        {
            return (vptr == 0) && (object == 0);
        }
        operator typename util::safe_bool<function_base>::result_type() const
        {
            return util::safe_bool<function_base>()(!empty());
        }
        bool operator!() const
        {
            return empty();
        }
        void reset()
        {
            if (!empty())
            {
                vptr->static_delete(&object);
                vptr = 0;
                object = 0;
            }
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16 , A17 a17 , A18 a18 , A19 a19 , A20 a20 , A21 a21 , A22 a22) const
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18 , a19 , a20 , a21 , a22);
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16 , A17 a17 , A18 a18 , A19 a19 , A20 a20 , A21 a21 , A22 a22)
        {
            BOOST_ASSERT(!empty());
            return vptr->invoke(&object , a0 , a1 , a2 , a3 , a4 , a5 , a6 , a7 , a8 , a9 , a10 , a11 , a12 , a13 , a14 , a15 , a16 , a17 , a18 , a19 , a20 , a21 , a22);
        }
    private:
        BOOST_COPYABLE_AND_MOVABLE(function_base);
    protected:
        vtable_ptr_type *vptr;
        mutable void *object;
    };
}}
