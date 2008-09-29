//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_MANAGED_COMPONENT_BASE_JUN_04_2008_0902PM)
#define HPX_COMPONENTS_MANAGED_COMPONENT_BASE_JUN_04_2008_0902PM

#include <boost/throw_exception.hpp>
#include <boost/noncopyable.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/detail/atomic_count.hpp>
#include <boost/type_traits/is_same.hpp>

#include <hpx/exception.hpp>
#include <hpx/runtime/components/server/wrapper_heap.hpp>
#include <hpx/runtime/components/server/wrapper_heap_list.hpp>
#include <hpx/util/static.hpp>

namespace hpx { namespace components 
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        struct this_type {};

        ///////////////////////////////////////////////////////////////////////
        template <typename HasUseCount>
        struct wrapper_use_count;

        template <>
        struct wrapper_use_count<boost::mpl::false_> : boost::noncopyable
        {
        };

        template <>
        struct wrapper_use_count<boost::mpl::true_> : boost::noncopyable
        {
            wrapper_use_count() 
              : use_count_(0) 
            {}

            boost::detail::atomic_count use_count_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \class managed_component_base managed_component_base.hpp hpx/runtime/components/server/managed_component_base.hpp
    ///
    /// The managed_component_base template is used as a indirection layer 
    /// for components allowing to gracefully handle the access to non-existing 
    /// components.
    ///
    /// Additionally it provides memory management capabilities for the 
    /// wrapping instances, and it integrates the memory management with the 
    /// DGAS service. Every instance of a managed_component_base gets assigned 
    /// a global id.
    /// The provided memory management allocates the managed_component_base 
    /// instances from a special heap, ensuring fast allocation and avoids a 
    /// full network round trip to the DGAS service for each of the allocated 
    /// instances.
    ///
    /// \tparam Component
    /// \tparam Derived
    /// \tparam HasUseCount
    ///
    template <
        typename Component, typename Derived = detail::this_type, 
        typename HasUseCount = boost::mpl::false_
    >
    class managed_component_base 
      : public detail::wrapper_use_count<HasUseCount>
    {
    private:
        typedef typename boost::mpl::if_<
                boost::is_same<Derived, detail::this_type>, 
                managed_component_base, Derived
            >::type derived_type;

    public:
        typedef Component wrapped_type;

        /// \brief Construct an empty managed_component_base
        managed_component_base() 
          : component_(0) 
        {}

        /// \brief Construct a managed_component_base instance holding a 
        ///        wrapped instance. This constructor takes ownership of the 
        ///        passed pointer.
        ///
        /// \param c    [in] The pointer to the wrapped instance. The 
        ///             managed_component_base takes ownership of this pointer.
        explicit managed_component_base(Component* c) 
          : component_(c) 
        {}

        /// \brief The destructor releases any wrapped instances
        ~managed_component_base()
        {
            delete component_;
            component_ = 0;
        }

        /// \brief Return a pointer to the wrapped instance
        Component* get()
        {
            if (0 == component_)
                boost::throw_exception(hpx::exception(invalid_status));
            return component_;
        }
        Component const* get() const
        {
            if (0 == component_)
                boost::throw_exception(hpx::exception(invalid_status));
            return component_;
        }

    protected:
        // the memory for the wrappers is managed by a one_size_heap_list
        typedef detail::wrapper_heap_list<
            detail::fixed_wrapper_heap<managed_component_base> > 
        heap_type;

        struct wrapper_heap_tag {};

        static heap_type& get_heap()
        {
            // ensure thread-safe initialization
            static util::static_<heap_type, wrapper_heap_tag> heap;
            return heap.get();
        }

    public:
        /// \brief  The memory for managed_component_base objects is managed by 
        ///         a class specific allocator. This allocator uses a one size 
        ///         heap implementation, ensuring fast memory allocation.
        ///         Additionally the heap registers the allocated  
        ///         managed_component_base instance with the DGAS service.
        ///
        /// \param size   [in] The parameter \a size is supplied by the 
        ///               compiler and contains the number of bytes to allocate.
        static void* operator new(std::size_t size)
        {
            if (size > sizeof(managed_component_base))
                return ::operator new(size);
            return get_heap().alloc();
        }
        static void operator delete(void* p, std::size_t size)
        {
            if (NULL == p) 
                return;     // do nothing if given a NULL pointer

            if (size != sizeof(managed_component_base)) {
                ::operator delete(p);
                return;
            }
            get_heap().free(p);
        }

        /// \brief  The placement operator new have to be overloaded as well 
        ///         (the global placement operators are hidden because of the 
        ///         new/delete overloads above).
        static void* operator new(std::size_t, void *p)
        {
            return p;
        }
        /// \brief  This operator delete is called only if the placement new 
        ///         fails.
        static void operator delete(void*, void*)
        {}

        /// \brief  The function \a create is used for allocation and 
        //          initialization of arrays of wrappers.
        static managed_component_base* create(std::size_t count)
        {
            // allocate the memory
            managed_component_base* p = get_heap().alloc(count);
            if (1 == count)
                return new (p) derived_type;

            // call constructors
            std::size_t succeeded = 0;
            try {
                derived_type* curr = static_cast<derived_type*>(p);
                for (std::size_t i = 0; i < count; ++i, ++curr) {
                    new (curr) derived_type;     // call placement new, might throw
                    ++succeeded;
                }
            }
            catch (...) {
                // call destructors for successfully constructed objects
                derived_type* curr = static_cast<derived_type*>(p);
                for (std::size_t i = 0; i < succeeded; ++i)
                    curr->~derived_type();
                get_heap().free(p, count);     // free memory
                throw;      // rethrow
            }
            return p;
        }
        /// \brief  The function \a destroy is used for deletion and 
        //          de-allocation of arrays of wrappers
        static void destroy(derived_type* p, std::size_t count)
        {
            if (NULL == p || 0 == count) 
                return;     // do nothing if given a NULL pointer

            if (1 == count) {
                p->~derived_type();
            }
            else {
                // call destructors for all managed_component_base instances
                derived_type* curr = static_cast<derived_type*>(p);
                for (std::size_t i = 0; i < count; ++i)
                    curr->~derived_type();
            }

            // free memory itself
            get_heap().free(p, count);
        }

    public:
        ///
        naming::id_type 
        get_gid(applier::applier& appl) const
        {
            return get_heap().get_gid(appl, 
                const_cast<managed_component_base*>(this));
        }

        ///////////////////////////////////////////////////////////////////////
        // The managed_component_base behaves just like the wrapped object
        Component* operator-> ()
        {
            if (0 == component_)
                boost::throw_exception(hpx::exception(invalid_status));
            return component_;
        }

        Component const* operator-> () const
        {
            if (0 == component_)
                boost::throw_exception(hpx::exception(invalid_status));
            return component_;
        }

        ///////////////////////////////////////////////////////////////////////
        Component& operator* ()
        {
            if (0 == component_)
                boost::throw_exception(hpx::exception(invalid_status));
            return *component_;
        }

        Component const& operator* () const
        {
            if (0 == component_)
                boost::throw_exception(hpx::exception(invalid_status));
            return *component_;
        }

        /// \brief Return the type of the embedded component
        static components::component_type get_type() 
        {
            return Component::get_component_type();
        }

    private:
        Component* component_;
    };

    // support for boost::intrusive_ptr<managed_component_base<...> >
    template <typename Component, typename Derived, typename HasUseCount>
    inline void
    intrusive_ptr_add_ref(managed_component_base<Component, Derived, HasUseCount>* p)
    {
        ++p->use_count_;
    }

    template <typename Component, typename Derived, typename HasUseCount>
    inline void
    intrusive_ptr_release(managed_component_base<Component, Derived, HasUseCount>* p)
    {
        if (--p->use_count_ == 0)
            delete p;
    }

}}

#endif


