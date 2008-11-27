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
#include <hpx/util/util.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Component, typename Wrapper = this_type>
        class managed_component_base : public managed_component_tag
        {
        private:
            static component_type value;

        public:
            // components must contain a typedef for wrapping_type defining the
            // managed_component type used to encapsulate instances of this 
            // component
            typedef managed_component<Component, Wrapper> wrapping_type;

            // This is the component id. Every component needs to have an embedded
            // enumerator 'value' which is used by the generic action implementation
            // to associate this component with a given action.
            static component_type get_component_type()
            {
                return value;
            }
            static void set_component_type(component_type type)
            {
                value = type;
            }

            /// \brief finalize() will be called just before the instance gets 
            ///        destructed
            ///
            /// \param self [in] The PX \a thread used to execute this function.
            /// \param appl [in] The applier to be used for finalization of the 
            ///             component instance. 
            void finalize() {}
        };

        ///////////////////////////////////////////////////////////////////////////
        template <typename Component, typename Wrapper>
        component_type managed_component_base<Component, Wrapper>::value = 
            component_invalid;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \class managed_component managed_component.hpp hpx/runtime/components/server/managed_component.hpp
    ///
    /// The managed_component template is used as a indirection layer 
    /// for components allowing to gracefully handle the access to non-existing 
    /// components.
    ///
    /// Additionally it provides memory management capabilities for the 
    /// wrapping instances, and it integrates the memory management with the 
    /// AGAS service. Every instance of a managed_component gets assigned 
    /// a global id.
    /// The provided memory management allocates the managed_component 
    /// instances from a special heap, ensuring fast allocation and avoids a 
    /// full network round trip to the AGAS service for each of the allocated 
    /// instances.
    ///
    /// \tparam Component
    /// \tparam Derived
    /// \tparam HasUseCount
    ///
    template <typename Component, typename Derived>
    class managed_component : boost::noncopyable
    {
    private:
        typedef typename boost::mpl::if_<
                boost::is_same<Derived, detail::this_type>, 
                managed_component, Derived
            >::type derived_type;

    public:
        typedef Component wrapped_type;
        typedef Component type_holder;

        /// \brief Construct an empty managed_component
        managed_component() 
          : component_(0) 
        {}

        /// \brief Construct a managed_component instance holding a 
        ///        wrapped instance. This constructor takes ownership of the 
        ///        passed pointer.
        ///
        /// \param c    [in] The pointer to the wrapped instance. The 
        ///             managed_component takes ownership of this pointer.
        explicit managed_component(Component* c) 
          : component_(c) 
        {}

        /// \brief Construct a managed_component instance holding a new wrapped
        ///        instance
        ///
        /// \param appl [in] The applier to be used for construction of the new
        ///             wrapped instance. 
        explicit managed_component(applier::applier& appl) 
          : component_(new wrapped_type(appl)) 
        {}

        /// \brief The destructor releases any wrapped instances
        ~managed_component()
        {
            delete component_;
            component_ = 0;
        }

        /// \brief finalize() will be called just before the instance gets 
        ///        destructed
        ///
        /// \param appl [in] The applier to be used for finalization of the 
        ///             component instance. 
        void finalize() 
        { 
            component_->finalize();
        }

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        static component_type get_component_type()
        {
            return wrapped_type::get_component_type();
        }
        static void set_component_type(component_type t)
        {
            wrapped_type::set_component_type(t);
        }

        /// \brief Return a pointer to the wrapped instance
        Component* get()
        {
            if (0 == component_) {
                HPX_OSSTREAM strm;
                strm << "component is NULL (" 
                     << components::get_component_type_name(get_component_type()) 
                     << ")";
                HPX_THROW_EXCEPTION(invalid_status, HPX_OSSTREAM_GETSTRING(strm));
            }
            return component_;
        }
        Component const* get() const
        {
            if (0 == component_) {
                HPX_OSSTREAM strm;
                strm << "component is NULL (" 
                     << components::get_component_type_name(get_component_type())
                     << ")";
                HPX_THROW_EXCEPTION(invalid_status, HPX_OSSTREAM_GETSTRING(strm));
            }
            return component_;
        }

    protected:
        // the memory for the wrappers is managed by a one_size_heap_list
        typedef detail::wrapper_heap_list<
            detail::fixed_wrapper_heap<managed_component> > 
        heap_type;

        struct wrapper_heap_tag {};

        static heap_type& get_heap()
        {
            // ensure thread-safe initialization
            util::static_<heap_type, wrapper_heap_tag> heap(get_component_type());
            return heap.get();
        }

    public:
        /// \brief  The memory for managed_component objects is managed by 
        ///         a class specific allocator. This allocator uses a one size 
        ///         heap implementation, ensuring fast memory allocation.
        ///         Additionally the heap registers the allocated  
        ///         managed_component instance with the AGAS service.
        ///
        /// \param size   [in] The parameter \a size is supplied by the 
        ///               compiler and contains the number of bytes to allocate.
        static void* operator new(std::size_t size)
        {
            if (size > sizeof(managed_component))
                return ::operator new(size);
            return get_heap().alloc();
        }
        static void operator delete(void* p, std::size_t size)
        {
            if (NULL == p) 
                return;     // do nothing if given a NULL pointer

            if (size != sizeof(managed_component)) {
                ::operator delete(p);
                return;
            }
            get_heap().free(p);
        }

        /// \brief  The placement operator new has to be overloaded as well 
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
        static managed_component* create(std::size_t count)
        {
            // allocate the memory
            managed_component* p = get_heap().alloc(count);
            if (1 == count)
                return new (p) derived_type();

            // call constructors
            std::size_t succeeded = 0;
            try {
                derived_type* curr = static_cast<derived_type*>(p);
                for (std::size_t i = 0; i < count; ++i, ++curr) {
                    // call placement new, might throw
                    new (curr) derived_type();
                    ++succeeded;
                }
            }
            catch (...) {
                // call destructors for successfully constructed objects
                derived_type* curr = static_cast<derived_type*>(p);
                for (std::size_t i = 0; i < succeeded; ++i)
                {
                    curr->finalize();
                    curr->~derived_type();
                }
                get_heap().free(p, count);     // free memory
                throw;      // rethrow
            }
            return p;
        }

        /// \brief  The function \a destroy is used for deletion and 
        //          de-allocation of arrays of wrappers
        static void destroy(derived_type* p, std::size_t count = 1)
        {
            if (NULL == p || 0 == count) 
                return;     // do nothing if given a NULL pointer

            if (1 == count) {
                p->finalize();
                p->~derived_type();
            }
            else {
                // call destructors for all managed_component instances
                derived_type* curr = static_cast<derived_type*>(p);
                for (std::size_t i = 0; i < count; ++i)
                {
                    curr->finalize();
                    curr->~derived_type();
                }
            }

            // free memory itself
            get_heap().free(p, count);
        }

        /// \brief  The function \a get_factory_properties is used to 
        ///         determine, whether instances of the derived component can 
        ///         be created in blocks (i.e. more than one instance at once). 
        ///         This function is used by the \a distributing_factory to 
        ///         determine a correct allocation strategy
        static factory_property get_factory_properties()
        {
            // components derived from this template can be allocated in blocks
            return factory_is_multi_instance;
        }

    public:
        ///
        naming::id_type get_gid() const
        {
            return get_heap().get_gid(const_cast<managed_component*>(this));
        }

        ///////////////////////////////////////////////////////////////////////
        // The managed_component behaves just like the wrapped object
        Component* operator-> ()
        {
            if (0 == component_) {
                HPX_OSSTREAM strm;
                strm << "component is NULL (" 
                     << components::get_component_type_name(get_component_type())
                     << ")";
                HPX_THROW_EXCEPTION(invalid_status, HPX_OSSTREAM_GETSTRING(strm));
            }
            return component_;
        }

        Component const* operator-> () const
        {
            if (0 == component_) {
                HPX_OSSTREAM strm;
                strm << "component is NULL (" 
                     << components::get_component_type_name(get_component_type())
                     << ")";
                HPX_THROW_EXCEPTION(invalid_status, HPX_OSSTREAM_GETSTRING(strm));
            }
            return component_;
        }

        ///////////////////////////////////////////////////////////////////////
        Component& operator* ()
        {
            if (0 == component_) {
                HPX_OSSTREAM strm;
                strm << "component is NULL (" 
                     << components::get_component_type_name(get_component_type())
                     << ")";
                HPX_THROW_EXCEPTION(invalid_status, HPX_OSSTREAM_GETSTRING(strm));
            }
            return *component_;
        }

        Component const& operator* () const
        {
            if (0 == component_) {
                HPX_OSSTREAM strm;
                strm << "component is NULL (" 
                     << components::get_component_type_name(get_component_type())
                     << ")";
                HPX_THROW_EXCEPTION(invalid_status, HPX_OSSTREAM_GETSTRING(strm));
            }
            return *component_;
        }

    protected:
        Component* component_;
    };

}}

#endif


