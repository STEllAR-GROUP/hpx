/*=============================================================================
    Copyright (c) 2007 Tobias Schwinger
  
    Use modification and distribution are subject to the Boost Software 
    License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
==============================================================================*/

#ifndef BOOST_UTILITY_MUTEXED_SINGLETON_HPP_INCLUDED
#   define BOOST_UTILITY_MUTEXED_SINGLETON_HPP_INCLUDED

#   include <boost/config.hpp>

#   ifdef BOOST_HAS_THREADS
#     include <boost/noncopyable.hpp>
#     include <boost/detail/workaround.hpp>
#     include <boost/aligned_storage.hpp>
#     include <boost/type_traits/alignment_of.hpp>
#     include <boost/thread/once.hpp>
#     include <boost/thread/recursive_mutex.hpp>
#     include <boost/utility/detail/restricted.hpp>
#     include <boost/utility/detail/member_dereference.hpp>
#     include <boost/utility/detail/singleton_manager.hpp>

namespace boost
{
    namespace detail
    {
        struct mutexed_singleton_context
        {
            detail::singleton_manager_context obj_mgr_context;
            boost::once_flag flg_initialized;
            boost::recursive_mutex* ptr_mutex;
        };
    }

    template< class Derived, int DisposalSlot = 0,
        typename SubsystemTag = void >
    class mutexed_singleton : boost::noncopyable
    {
        class instance_proxy; 
      public:

        static instance_proxy const instance;
        class lease;

      protected:

        inline mutexed_singleton()
        { 
            // enforce instantiation to ensure proper ctor
            static_cast<void>(& instance_proxy::create_instance);
        }

        inline ~mutexed_singleton()
        {
            detail::singleton_manager<SubsystemTag>::again(
                instance_proxy::obj_context.flg_initialized);
        }

        typedef mutexed_singleton<Derived,DisposalSlot,
            SubsystemTag> base_class_type;

        typedef detail::mutexed_singleton_context& context_type;

        friend struct detail::singleton_initialization;
    };

    template< class Derived, int DisposalSlot, typename SubsystemTag >
    class mutexed_singleton<Derived,DisposalSlot,SubsystemTag>::lease
#     if BOOST_WORKAROUND(BOOST_MSVC, <= 1400)
      // friends don't work correctly prior to VS 2005 SP1
      : protected mutexed_singleton<Derived,DisposalSlot,SubsystemTag>
          ::instance_proxy
#     endif
    {
        detail::mutexed_singleton_context* ptr_context;
        mutable boost::recursive_mutex::scoped_lock lock;
      public:

        inline lease()
          : ptr_context( & detail::singleton_initialization::call<Derived,
                mutexed_singleton<Derived,DisposalSlot,SubsystemTag> >() )
          , lock(*ptr_context->ptr_mutex)
        { }

        inline lease(lease const & that)
          : ptr_context(that.ptr_context)
          , lock(*ptr_context->ptr_mutex)
        { }

        inline Derived* operator->() const
        {
            return static_cast<Derived*>(ptr_context->obj_mgr_context.ptr_that);
        }

        template< typename MP >
        inline typename detail::member_dereference<Derived,MP>::type
        operator->*(MP mp) const
        {
            return detail::member_dereference<Derived,MP>(static_cast<Derived*>(
                ptr_context->obj_mgr_context.ptr_that), mp);
        }

        friend lease get_pointer(lease const& that)
        {
            return lease(that);
        }
    };

    template< class Derived, int DisposalSlot, typename SubsystemTag >
    class mutexed_singleton<Derived,DisposalSlot,SubsystemTag>::instance_proxy
    {
#     if BOOST_WORKAROUND(BOOST_MSVC, <= 1400)
      protected:
#     endif
        static detail::mutexed_singleton_context obj_context;

        template< typename MP >
        struct member_deref_proxy
          : private lease, public detail::member_dereference<Derived,MP>
        {
            member_deref_proxy(MP mp)
              : lease(), detail::member_dereference<Derived,MP>(
                    lease::operator->(), mp)
            { }
        };

#     if !BOOST_WORKAROUND(BOOST_MSVC, BOOST_TESTED_AT(1400))

      public:

        inline lease operator->() const
        {
            return lease();
        }
#     else // strange workaround to keep MSVC from ICEing:

        struct lease_ : lease { };

      public:

        inline lease_ operator->() const
        {
            return lease_();
        }

#     endif

        template< typename MP >
        inline typename member_deref_proxy<MP>::type operator->*(MP mp) const
        {
            return member_deref_proxy<MP>(mp);
        }

        friend instance_proxy const& get_pointer(instance_proxy const& that)
        {
            return that;
        }

      private:
#     if BOOST_WORKAROUND(BOOST_MSVC, <= 1400)
      protected:
#     endif

        inline static detail::mutexed_singleton_context& init()
        {
            boost::call_once(& create_instance, obj_context.flg_initialized);
            return obj_context;
        }

        static void create_instance()
        {
            static typename boost::aligned_storage< sizeof(Derived),
                ::boost::alignment_of<Derived>::value >::type buf_instance;

            obj_context.obj_mgr_context.ptr_that =  new (& buf_instance)
                Derived( detail::restricted_argument() );

            static typename boost::aligned_storage< sizeof(boost::recursive_mutex),
                ::boost::alignment_of<boost::recursive_mutex>::value 
                >::type buf_mutex;

            obj_context.ptr_mutex =  new (& buf_mutex) boost::recursive_mutex();

            detail::singleton_manager<SubsystemTag>::attach(
                obj_context.obj_mgr_context);
        }

        friend class lease;
        template< class D, int DS, typename ST > friend class mutexed_singleton;

        friend struct detail::singleton_initialization;

        instance_proxy() { }
        instance_proxy(instance_proxy const &) { }
    };

    template< class Derived, int DisposalSlot, typename SubsystemTag >
    detail::mutexed_singleton_context
        mutexed_singleton<Derived,DisposalSlot,SubsystemTag>::instance_proxy
            ::obj_context = 
        { BOOST_DETAIL_SINGLETON_CONTEXT_INIT(call_dtor,Derived,DisposalSlot)
        , BOOST_ONCE_INIT, 0l
        };

    template< class Derived, int DisposalSlot, typename SubsystemTag >
    typename mutexed_singleton<Derived,DisposalSlot,SubsystemTag>
        ::instance_proxy const
            mutexed_singleton<Derived,DisposalSlot,SubsystemTag>::instance
                = typename mutexed_singleton<Derived,DisposalSlot>::
                    instance_proxy();

#   else // !defined(BOOST_HAS_THREADS)

#     include <boost/utility/singleton.hpp>

namespace boost
{
    // In a single threaded environment a simple singleton behaves as if it
    // was mutexed
    template< class Derived, int DisposalSlot = 0,
        typename SubsystemTag = void >
    class mutexed_singleton
      : public boost::singleton<Derived,DisposalSlot,SubsystemTag>
    {
      protected:

        mutexed_singleton()
        { }
    };

#   endif 

} // namespace boost

#endif // include guard

