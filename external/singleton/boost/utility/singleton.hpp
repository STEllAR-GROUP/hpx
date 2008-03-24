/*=============================================================================
    Copyright (c) 2007 Tobias Schwinger
  
    Use modification and distribution are subject to the Boost Software 
    License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
==============================================================================*/

#ifndef BOOST_UTILITY_SINGLETON_HPP_INCLUDED
#   define BOOST_UTILITY_SINGLETON_HPP_INCLUDED

#   include <boost/config.hpp>
#   include <boost/detail/workaround.hpp>
#   include <boost/noncopyable.hpp>
#   include <boost/aligned_storage.hpp>
#   include <boost/type_traits/alignment_of.hpp>
#   include <boost/utility/detail/restricted.hpp>
#   include <boost/utility/detail/member_dereference.hpp>
#   include <boost/utility/detail/singleton_manager.hpp>

#   ifdef BOOST_HAS_THREADS
#       include <boost/thread/once.hpp>
#   endif

namespace boost
{
    namespace detail { struct singleton_context; }

    template< class Derived, int DisposalSlot = 0, 
        typename SubsystemTag = void >
    class singleton : boost::noncopyable
    {
        class instance_proxy;
      public:

        static instance_proxy const instance;
        class lease;

      protected:

        inline singleton()
        {
            // enforce instantiation to ensure proper ctor
            static_cast<void>(& instance_proxy::create_instance);
        }

        inline ~singleton()
        {
#   ifdef BOOST_HAS_THREADS
            detail::singleton_manager<SubsystemTag>::again(
                instance_proxy::obj_context.flg_initialized);
#   else
            instance_proxy::obj_context.obj_mgr_context.ptr_that = 0l;
#   endif
        }

        typedef singleton<Derived,DisposalSlot,SubsystemTag> base_class_type;
        typedef detail::singleton_context context_type;


        friend struct detail::singleton_initialization;
    };

    namespace detail
    {
        struct singleton_context
        {
            detail::singleton_manager_context obj_mgr_context;
#   ifdef BOOST_HAS_THREADS
            boost::once_flag flg_initialized;
#   endif
        };
    }

    template< class Derived, int DisposalSlot, typename SubsystemTag >
    class singleton<Derived,DisposalSlot,SubsystemTag>::instance_proxy
    {
#   if BOOST_WORKAROUND(BOOST_MSVC, <= 1400)
        // friends don't work correctly prior to VS 2005 SP1
      protected:
#   endif
        static detail::singleton_context obj_context;
      public:

        inline Derived* operator->() const
        {
            return static_cast<Derived*>(
                detail::singleton_initialization::call<Derived,
                    singleton<Derived,DisposalSlot,SubsystemTag> >()
                .obj_mgr_context.ptr_that );
        }

        template< typename MP >
        inline typename detail::member_dereference<Derived,MP>::type 
        operator->*(MP mp) const
        {
            return detail::member_dereference<Derived,MP>(
                static_cast<Derived*>(
                    detail::singleton_initialization::call<Derived,
                        singleton<Derived,DisposalSlot,SubsystemTag> >()
                    .obj_mgr_context.ptr_that), 
                mp);
        }

        friend Derived* get_pointer(instance_proxy const & x)
        {
            return x.operator->();
        }

      private:
#   if BOOST_WORKAROUND(BOOST_MSVC, <= 1400)
      protected:
#   endif

        inline static detail::singleton_context& init()
        {
#   ifdef BOOST_HAS_THREADS
            boost::call_once(& create_instance, obj_context.flg_initialized);
#   else
            if (! obj_context.obj_mgr_context.ptr_that) create_instance();
#   endif
            return obj_context;
        }

        static void create_instance()
        {
            static typename boost::aligned_storage< sizeof(Derived),
                ::boost::alignment_of<Derived>::value >::type buf_instance;

            obj_context.obj_mgr_context.ptr_that = new (& buf_instance) Derived(
                detail::restricted_argument() );

            detail::singleton_manager<SubsystemTag>::attach(
                obj_context.obj_mgr_context);
        }

        template< class D, int DS, typename ST > friend class singleton;
        friend class lease;

        friend struct detail::singleton_initialization;

        instance_proxy() { }
        instance_proxy(instance_proxy const &) { }
    };

    template< class Derived, int DisposalSlot, typename SubsystemTag >
    class singleton<Derived,DisposalSlot,SubsystemTag>::lease
#   if BOOST_WORKAROUND(BOOST_MSVC, <= 1400)
      : singleton<Derived,DisposalSlot,SubsystemTag>::instance_proxy
#   endif
    {
        Derived* ptr_that;
      public:

        inline lease()
          : ptr_that( static_cast<Derived*>(
              detail::singleton_initialization::call<Derived,
                  singleton<Derived,DisposalSlot,SubsystemTag> >()
              .obj_mgr_context.ptr_that ))
        {
        }

        inline Derived* operator->() const
        {
            return this->ptr_that;
        }

        template< typename MP >
        inline typename detail::member_dereference<Derived,MP>::type
        operator->*(MP mp) const
        {
            return detail::member_dereference<Derived,MP>(this->ptr_that,mp);
        }

        friend Derived* get_pointer(lease const& x)
        {
            return x.operator->();
        }
    };

    template< class Derived, int DisposalSlot, typename SubsystemTag >
    detail::singleton_context
        singleton<Derived,DisposalSlot,SubsystemTag>::instance_proxy
            ::obj_context = 
        {
            BOOST_DETAIL_SINGLETON_CONTEXT_INIT(call_dtor,Derived,DisposalSlot)
#   ifdef BOOST_HAS_THREADS
          , BOOST_ONCE_INIT
#   endif 
        };

    template< class Derived, int DisposalSlot, typename SubsystemTag >
    typename singleton<Derived,DisposalSlot,SubsystemTag>::instance_proxy const
        singleton<Derived,DisposalSlot,SubsystemTag>::instance = 
            typename singleton<Derived,DisposalSlot,SubsystemTag>::
                instance_proxy();

} // namespace boost

#endif

