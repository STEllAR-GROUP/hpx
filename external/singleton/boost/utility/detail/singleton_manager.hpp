/*=============================================================================
    Copyright (c) 2007 Tobias Schwinger
  
    Use modification and distribution are subject to the Boost Software 
    License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
==============================================================================*/

#ifndef BOOST_UTILITY_DETAIL_SINGLETON_MANAGER_HPP_INCLUDED
#   define BOOST_UTILITY_DETAIL_SINGLETON_MANAGER_HPP_INCLUDED

#   include <new>
#   include <boost/config.hpp>
#   include <boost/mpl/bool.hpp>

#   ifdef BOOST_HAS_THREADS
#     include <cstring>
#     include <boost/thread/once.hpp>
#     include <boost/thread/mutex.hpp>
#   endif

#   if defined(__GNUC__) && __GNUC__ >= 4
#     pragma GCC visibility push(hidden)
#   endif

namespace boost 
{ 
    namespace detail
    {
        struct singleton_manager_context
        {
            void*                       ptr_that;
            void                      (*fnc_dtor)(singleton_manager_context*);
            int                         val_slot;
            singleton_manager_context*  ptr_next;

            // templates to initialize fnc_dtor with:

            template< typename T >
            static void call_dtor(singleton_manager_context* ctx)
            {
                static_cast<T*>(ctx->ptr_that) -> ~T();
            }

            template< typename T >
            static void call_delete(singleton_manager_context* ctx)
            {
                delete static_cast<T*>(ctx->ptr_that);
            }
        };

#   define BOOST_DETAIL_SINGLETON_CONTEXT_INIT(disposal,type,slot) \
    { 0l, & detail::singleton_manager_context:: disposal < type >, slot, 0l }

        template< typename Tag >
        class singleton_manager
        {
            typedef singleton_manager_context context;
            context* volatile ptr_first;
#   ifdef BOOST_HAS_THREADS
            boost::mutex obj_mutex;
#   endif

            static singleton_manager<Tag>* ptr_instance;

            struct destruction_sensor
            {
                ~destruction_sensor() { singleton_manager<Tag>::cleanup(); }
            };
            static destruction_sensor const obj_destruction_sensor;

            static void create_instance()
            {
                static typename boost::aligned_storage<
                    sizeof(singleton_manager),
                    ::boost::alignment_of< singleton_manager<Tag> >::value 
                    >::type buf_instance;

                ptr_instance = new (& buf_instance) singleton_manager<Tag>();
            }

            inline singleton_manager() : ptr_first(0l) { }

          public:

            inline void insert_context(context& d)
            {
                context*volatile* hook = & this->ptr_first;
                context* next = *hook;
                for (; !!next && next->val_slot < d.val_slot; next = *hook)
                    hook = & next->ptr_next;
                d.ptr_next = next;
                *hook = & d;
            }

            static inline void attach(context& d)
            {
#   ifdef BOOST_HAS_THREADS
                static boost::once_flag initialized = BOOST_ONCE_INIT;
                boost::call_once(& create_instance, initialized);
                boost::mutex::scoped_lock lock(ptr_instance->obj_mutex); 
#   else
                if (! ptr_instance) create_instance();
#   endif
                ptr_instance->insert_context(d);
            }

            static inline void cleanup()
            {
                if (!!ptr_instance)
                { 
                    context* i;
                    while (!!(i = ptr_instance->ptr_first))
                    {
                        context* next = i->ptr_next;
                        i->fnc_dtor(i);
                        ptr_instance->ptr_first = next;
                    }
                }
            }

#   ifdef BOOST_HAS_THREADS
            static inline void again(boost::once_flag& of)
            {
                static boost::once_flag uninitialized = BOOST_ONCE_INIT;
                std::memcpy(& of, & uninitialized, sizeof(boost::once_flag));
            }
#   endif
        };

        template< typename Tag >
        singleton_manager<Tag>* singleton_manager<Tag>::ptr_instance = 0l; 

        template< typename Tag >
        typename singleton_manager<Tag>::destruction_sensor const
            singleton_manager<Tag>::obj_destruction_sensor = destruction_sensor();

        //

        struct singleton_initialization
        {
            struct udc { template< typename T > udc(T const&); };
            template< class T, void(*X)() > struct has_member;

            template< class T > static char has_placement_tester(
                has_member<T,& T::singleton_placement>*);
            template< class T > static char (& has_placement_tester(udc))[2];

            template< class C, class Base >
            static inline typename Base::context_type call_impl(
                mpl::bool_<true> const&)
            {
                return C::singleton_placement();
            } 
            template< class C, class Base >
            static inline typename Base::context_type call_impl(
                mpl::bool_<false> const&)
            {
                return Base::instance_proxy::init();
            }

            template< class C, class Base >
            static inline typename Base::context_type call()
            {
                return singleton_initialization::call_impl<C,Base>(
                    mpl::bool_< 1 == sizeof(
                        has_placement_tester<C>(0l)) >() );
            }
        };

    } // namespace detail

    template< typename SubsystemTag >
    inline void destroy_singletons()
    {
        detail::singleton_manager<SubsystemTag>::cleanup();
    }

    inline void destroy_singletons()
    {
        detail::singleton_manager<void>::cleanup();
    }

#   if defined(__GNUC__) && __GNUC__ >= 4
#     pragma GCC visibility pop
#   endif

#   define BOOST_SINGLETON_PLACEMENT_DECLARATION \
        friend struct boost::detail::singleton_initialization; \
        static context_type singleton_placement();

#   define BOOST_SINGLETON_PLACEMENT(Type) \
        Type::context_type Type::singleton_placement() \
        { \
            typedef Type::base_class_type b; \
            return boost::detail::singleton_initialization::call_impl<void,b>( \
                boost::mpl::bool_<false>() ); \
        }

} // namespace boost

#endif 

