//  Copyright (c) 2007-2011 Hartmut Kaiser
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
#include <boost/preprocessor/repeat.hpp>

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/wrapper_heap.hpp>
#include <hpx/runtime/components/server/wrapper_heap_list.hpp>
#include <hpx/util/static.hpp>
#include <hpx/util/stringstream.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components 
{
    template <typename Component, typename Derived>
    class managed_component;

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Wrapper>
    class managed_component_base 
      : public detail::managed_component_tag, boost::noncopyable
    {
    public:
        // components must contain a typedef for wrapping_type defining the
        // managed_component type used to encapsulate instances of this 
        // component
        typedef managed_component<Component, Wrapper> wrapping_type;
        typedef Component base_type_holder;

        /// \brief finalize() will be called just before the instance gets 
        ///        destructed
        void finalize() {}

        // This exposes the component type
        static component_type get_component_type()
        {
            return components::get_component_type<Component>();
        }
        static void set_component_type(component_type type)
        {
            components::set_component_type<Component>(type);
        }

        template <typename ManagedType>
        naming::id_type const& get_gid(ManagedType* p) const
        {
            if (!id_) 
                id_ = naming::id_type(p->get_base_gid(), naming::id_type::unmanaged);
            return id_;
        }

        naming::id_type const& get_gid() const;

        naming::gid_type get_base_gid() const;

    private:
        template <typename, typename>
        friend class managed_component;

        mutable naming::id_type id_;
        managed_component<Component, Wrapper>* back_ptr_;        
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // for backwards compatibility only
        using components::managed_component_base;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Derived>
    struct heap_factory
    {
        // the memory for the wrappers is managed by a one_size_heap_list
        typedef detail::wrapper_heap_list<
            detail::fixed_wrapper_heap<Derived> > heap_type;

        typedef detail::fixed_wrapper_heap<Derived> block_type;

        struct wrapper_heap_tag {};

        static heap_type& get_heap()
        {
            // ensure thread-safe initialization
            util::static_<heap_type, wrapper_heap_tag, HPX_RUNTIME_INSTANCE_LIMIT> 
                heap(components::get_component_type<Component>());
            return heap.get(get_runtime_instance_number());
        }

        static block_type* alloc_heap()
        {
            return get_heap().alloc_heap();
        }

        static void add_heap(block_type* p)
        {
            return get_heap().add_heap(p);
        }

        static Derived* alloc(std::size_t count = 1)
        {
            return get_heap().alloc(count);
        }
        static void free(void* p, std::size_t count = 1)
        {
            get_heap().free(p, count);
        }
        static naming::gid_type get_gid(void* p)
        {
            return get_heap().get_gid(p);
        }
    };

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
        typedef typename Component::base_type_holder base_type_holder;

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
        managed_component() 
          : component_(new wrapped_type()) 
        {
            component_->back_ptr_ = this; 
        }

#define MANAGED_COMPONENT_CONSTRUCT(Z, N, _)                                  \
        template <BOOST_PP_ENUM_PARAMS(N, typename T)>                        \
        managed_component(BOOST_PP_ENUM_BINARY_PARAMS(N, T, const& t))        \
          : component_(new wrapped_type(BOOST_PP_ENUM_PARAMS(N, t)))          \
        {                                                                     \
            component_->back_ptr_ = this;                                     \
        }
    /**/

        BOOST_PP_REPEAT_FROM_TO(1, HPX_COMPONENT_CREATE_ARG_MAX, 
            MANAGED_COMPONENT_CONSTRUCT, _)

#undef MANAGED_COMPONENT_CONSTRUCT

        /// \brief The destructor releases any wrapped instances
        ~managed_component()
        {
            component_->finalize();
            delete component_;
            component_ = 0;
        }

        /// \brief finalize() will be called just before the instance gets 
        ///        destructed
        void finalize() {}  // finalize the wrapped component in our destructor

        static component_type get_component_type()
        {
            return components::get_component_type<wrapped_type>();
        }
        static void set_component_type(component_type t)
        {
            components::set_component_type<wrapped_type>(t);
        }

        /// \brief Return a pointer to the wrapped instance
        /// \note  Caller must check validity of returned pointer
        Component* get()
        {
            return component_;
        }
        Component const* get() const
        {
            return component_;
        }

        Component* get_checked()
        {
            if (0 == component_) {
                hpx::util::osstream strm;
                strm << "component is NULL (" 
                     << components::get_component_type_name(
                        components::get_component_type<wrapped_type>())
                     << ")";
                HPX_THROW_EXCEPTION(invalid_status, 
                    "managed_component<Component, Derived>::get_checked", 
                    hpx::util::osstream_get_string(strm));
            }
            return component_;
        }

        Component const* get_checked() const
        {
            if (0 == component_) {
                hpx::util::osstream strm;
                strm << "component is NULL (" 
                     << components::get_component_type_name(
                        components::get_component_type<wrapped_type>())
                     << ")";
                HPX_THROW_EXCEPTION(invalid_status, 
                    "managed_component<Component, Derived>::get_checked", 
                    hpx::util::osstream_get_string(strm));
            }
            return component_;
        }

    protected:
        typedef heap_factory<Component, derived_type> heap_type;

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
            derived_type* p = heap_type::alloc();
            if (NULL == p) 
                boost::throw_exception(std::bad_alloc());
            return p;
        }
        static void operator delete(void* p, std::size_t size)
        {
            if (NULL == p) 
                return;     // do nothing if given a NULL pointer

            if (size != sizeof(managed_component)) {
                ::operator delete(p);
                return;
            }
            heap_type::free(p);
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
        static derived_type* create(std::size_t count)
        {
            // allocate the memory
            derived_type* p = heap_type::alloc(count);
            if (NULL == p) 
                boost::throw_exception(std::bad_alloc());

            if (1 == count)
                return new (p) derived_type();

            // call constructors
            std::size_t succeeded = 0;
            try {
                derived_type* curr = p;
                for (std::size_t i = 0; i < count; ++i, ++curr) {
                    // call placement new, might throw
                    new (curr) derived_type();
                    ++succeeded;
                }
            }
            catch (...) {
                // call destructors for successfully constructed objects
                derived_type* curr = p;
                for (std::size_t i = 0; i < succeeded; ++i)
                {
                    curr->finalize();
                    curr->~derived_type();
                }
                heap_type::free(p, count);     // free memory
                throw;      // rethrow
            }
            return p;
        }

        /// \brief  The function \a create is used for allocation and 
        //          initialization of a single instance.
#define HPX_MANAGED_COMPONENT_CREATE_ONE(Z, N, _)                             \
        template <BOOST_PP_ENUM_PARAMS(N, typename T)>                        \
        static derived_type*                                                  \
        create_one(BOOST_PP_ENUM_BINARY_PARAMS(N, T, const& t))               \
        {                                                                     \
            derived_type* p = heap_type::alloc();                             \
            if (NULL == p) boost::throw_exception(std::bad_alloc());          \
            return new (p) derived_type(BOOST_PP_ENUM_PARAMS(N, t));          \
        }                                                                     \
    /**/

        BOOST_PP_REPEAT_FROM_TO(1, HPX_COMPONENT_CREATE_ARG_MAX, 
            HPX_MANAGED_COMPONENT_CREATE_ONE, _)

#undef HPX_MANAGED_COMPONENT_CREATE_ONE

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
                derived_type* curr = p;
                for (std::size_t i = 0; i < count; ++i)
                {
                    curr->finalize();
                    curr->~derived_type();
                }
            }

            // free memory itself
            heap_type::free(p, count);
        }

        /// \brief  The function \a get_factory_properties is used to 
        ///         determine, whether instances of the derived component can 
        ///         be created in blocks (i.e. more than one instance at once). 
        ///         This function is used by the \a distributing_factory to 
        ///         determine a correct allocation strategy
        static factory_property get_factory_properties()
        {
            // components derived from this template can be allocated in blocks
            return factory_none; // factory_is_multi_instance;
        }

    public:
        ///
        /// \brief Return the global id of this \a future instance
        naming::id_type const& get_gid() const
        {
            return get_checked()->get_gid(this);
        }

        ///////////////////////////////////////////////////////////////////////
        // The managed_component behaves just like the wrapped object
        Component* operator-> ()
        {
            return get_checked();
        }

        Component const* operator-> () const
        {
            return get_checked();
        }

        ///////////////////////////////////////////////////////////////////////
        Component& operator* ()
        {
            return *get_checked();
        }

        Component const& operator* () const
        {
            return *get_checked();
        }

        ///////////////////////////////////////////////////////////////////////
        naming::gid_type get_base_gid() const
        {
            return heap_type::get_gid(const_cast<managed_component*>(this));
        }

    protected:
        Component* component_;
    };

    template <typename Component, typename Wrapper>
    naming::id_type const&
    managed_component_base<Component, Wrapper>::get_gid() const
    {
        BOOST_ASSERT(back_ptr_);
        return get_gid(back_ptr_);
    } 

    template <typename Component, typename Wrapper>
    naming::gid_type 
    managed_component_base<Component, Wrapper>::get_base_gid() const
    {
        return get_gid().get_gid();
    } 
}}

#endif


