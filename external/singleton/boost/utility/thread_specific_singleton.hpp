/*=============================================================================
    Copyright (c) 2007 Tobias Schwinger
  
    Use modification and distribution are subject to the Boost Software 
    License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
    http://www.boost.org/LICENSE_1_0.txt).
==============================================================================*/

#ifndef BOOST_UTILITY_THREAD_SPECIFIC_SINGLETON_HPP_INCLUDED
#   define BOOST_UTILITY_THREAD_SPECIFIC_SINGLETON_HPP_INCLUDED

#   include <boost/config.hpp>
#   ifdef BOOST_HAS_THREADS
#     include <boost/noncopyable.hpp>
#     include <boost/detail/workaround.hpp>
#     include <boost/aligned_storage.hpp>
#     include <boost/type_traits/alignment_of.hpp>
#     include <boost/thread/once.hpp>
#     include <boost/thread/tss.hpp>
#     include <boost/utility/detail/restricted.hpp>
#     include <boost/utility/detail/member_dereference.hpp>
#     include <boost/utility/detail/singleton_manager.hpp>

namespace boost
{
    template< class Derived, int DisposalSlot = 0,
        typename SubsystemTag = void >
    class thread_specific_singleton : boost::noncopyable
    {
        class instance_proxy;
      public:

        static instance_proxy const instance;
        class lease;

      protected:

        inline thread_specific_singleton()
        { 
            // enforce instantiation to ensure proper ctor
            static_cast<void>(& instance_proxy::get);
        }

        inline ~thread_specific_singleton()
        {
            instance_proxy::ptr_tsp->release();
        }

        typedef void* context_type;

        friend struct detail::singleton_initialization;
    };

    template< class Derived, int DisposalSlot, typename SubsystemTag >
    class thread_specific_singleton<Derived,DisposalSlot,SubsystemTag>
        ::instance_proxy
    {
#     if BOOST_WORKAROUND(BOOST_MSVC, <= 1400)
      // friends don't work correctly prior to VS 2005 SP1
      protected:
#     endif
        struct holder;
        typedef boost::thread_specific_ptr<holder> ts_ptr;
        static ts_ptr* ptr_tsp;
      public:

        inline Derived* operator->() const
        {
            return get();
        }

        template< typename MP >
        inline typename detail::member_dereference<Derived,MP>::type
        operator->*(MP mp) const
        {
            return detail::member_dereference<Derived,MP>(get(),mp);
        }

        friend Derived* get_pointer(instance_proxy const &)
        {
            return get();
        }

      private:
#     if BOOST_WORKAROUND(BOOST_MSVC, <= 1400)
      protected:
#     endif

        struct holder
        {
            detail::singleton_manager_context obj_context;
            Derived obj_instance;

            holder()
              : obj_instance(detail::restricted_argument())
            {
                obj_context.ptr_that = this;
                obj_context.fnc_dtor =
                    & detail::singleton_manager_context::call_delete<holder>;
                obj_context.val_slot = DisposalSlot;
            } 
        };

        static void no_delete(holder*) { }

        inline static void* init()
        {
            static boost::once_flag flg_initialized = BOOST_ONCE_INIT;
            boost::call_once(& create_tsp, flg_initialized);
            holder* result = ptr_tsp->get();
            if (! result)
            {
                ptr_tsp->reset((result = new holder()));
                detail::singleton_manager<SubsystemTag>::attach(
                    result->obj_context);
            }
            return result;
        }

        inline static Derived* get()
        {
            return & reinterpret_cast<holder*>(
                detail::singleton_initialization::call< Derived,
                    thread_specific_singleton<Derived,DisposalSlot,SubsystemTag>
                >())->obj_instance;
        }

        static void create_tsp()
        {
            static typename boost::aligned_storage< sizeof(ts_ptr),
                ::boost::alignment_of<ts_ptr>::value >::type buf_tsp;

            ptr_tsp = new (& buf_tsp) ts_ptr(& no_delete);
        }

        template< class D, int DS, typename ST >
        friend class thread_specific_singleton;

        friend class lease;
        friend struct detail::singleton_initialization;

        instance_proxy() { }
        instance_proxy(instance_proxy const &) { }
    };

    template< class Derived, int DisposalSlot, typename SubsystemTag >
    class thread_specific_singleton<Derived,DisposalSlot,SubsystemTag>
      ::lease
#     if BOOST_WORKAROUND(BOOST_MSVC, <= 1400)
      : thread_specific_singleton<Derived,DisposalSlot,SubsystemTag>
          ::instance_proxy
#     endif
    {
        Derived* ptr_instance;
      public:

        inline lease()
            : ptr_instance(instance_proxy::get())
        { }

        inline Derived* operator->() const
        {
            return this->ptr_instance;
        }

        template< typename MP >
        inline typename detail::member_dereference<Derived,MP>::type
        operator->*(MP mp) const
        {
            return detail::member_dereference<Derived,MP>(
                this->ptr_instance,mp);
        }

        friend Derived* get_pointer(lease const& that)
        {
            return that.ptr_instance;
        }
    };

    template< class Derived, int DisposalSlot, typename SubsystemTag >
    typename thread_specific_singleton<Derived,DisposalSlot,SubsystemTag>
        ::instance_proxy::ts_ptr* 
            thread_specific_singleton<Derived,DisposalSlot,SubsystemTag>::
                instance_proxy::ptr_tsp = 0l;

    template< class Derived, int DisposalSlot, typename SubsystemTag >
    typename thread_specific_singleton<Derived,DisposalSlot,SubsystemTag>
        ::instance_proxy const
            thread_specific_singleton<Derived,DisposalSlot,SubsystemTag>
                ::instance = typename thread_specific_singleton<Derived,
                    DisposalSlot,SubsystemTag>::instance_proxy();

#   else // !defined(BOOST_HAS_THREADS)

#     include <boost/utility/singleton.hpp>

namespace boost
{

    // In a single threaded environment a simple singleton behaves as if
    // it was thread-specific
    template< class Derived, int DisposalSlot = 0,
        typename SubsystemTag = void > 
    class thread_specific_singleton
        : public boost::singleton<Derived,DisposalSlot,SubsystemTag>
    {
      protected:

        inline thread_specific_singleton()
        { }
    };

#   endif

} // namespace boost

#endif // include guard

