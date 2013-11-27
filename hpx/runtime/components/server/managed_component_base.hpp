//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_MANAGED_COMPONENT_BASE_JUN_04_2008_0902PM)
#define HPX_COMPONENTS_MANAGED_COMPONENT_BASE_JUN_04_2008_0902PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/wrapper_heap.hpp>
#include <hpx/runtime/components/server/wrapper_heap_list.hpp>
#include <hpx/util/reinitializable_static.hpp>

#include <boost/throw_exception.hpp>
#include <boost/noncopyable.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/detail/atomic_count.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/move/move.hpp>

#include <stdexcept>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    template <typename Component>
    struct managed_component_ctor_policy<
        Component, typename Component::has_managed_component_base>
    {
        typedef typename Component::ctor_policy type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component>
    struct managed_component_dtor_policy<
        Component, typename Component::has_managed_component_base>
    {
        typedef typename Component::dtor_policy type;
    };
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    template <typename Component, typename Derived>
    class managed_component;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail_adl_barrier
    {
        template <typename BackPtrTag>
        struct init;

        template <>
        struct init<traits::construct_with_back_ptr>
        {
            template <typename Component, typename Managed>
            static void call(Component* component, Managed* this_)
            {
            }

            template <typename Component, typename Managed>
            static void call_new(Component*& component, Managed* this_)
            {
                typedef typename Managed::wrapped_type wrapped_type;
                component = new wrapped_type(this_);
            }

#define MANAGED_COMPONENT_CONSTRUCT_INIT1(Z, N, _)                            \
            template <typename Component, typename Managed,                   \
                BOOST_PP_ENUM_PARAMS(N, typename T)>                          \
            static void call(Component*& component, Managed* this_,           \
                HPX_ENUM_FWD_ARGS(N, T, t))                                   \
            {                                                                 \
                typedef typename Managed::wrapped_type wrapped_type;          \
                component = new wrapped_type(this_,                           \
                    HPX_ENUM_FORWARD_ARGS(N , T, t));                         \
            }                                                                 \
    /**/
            BOOST_PP_REPEAT_FROM_TO(1, HPX_COMPONENT_CREATE_ARGUMENT_LIMIT,
                MANAGED_COMPONENT_CONSTRUCT_INIT1, _)

#undef MANAGED_COMPONENT_CONSTRUCT_INIT1
        };

        template <>
        struct init<traits::construct_without_back_ptr>
        {
            template <typename Component, typename Managed>
            static void call(Component* component, Managed* this_)
            {
                component->set_back_ptr(this_);
            }

            template <typename Component, typename Managed>
            static void call_new(Component*& component, Managed* this_)
            {
                typedef typename Managed::wrapped_type wrapped_type;
                component = new wrapped_type();
                component->set_back_ptr(this_);
            }

#define MANAGED_COMPONENT_CONSTRUCT_INIT2(Z, N, _)                            \
            template <typename Component, typename Managed,                   \
                BOOST_PP_ENUM_PARAMS(N, typename T)>                          \
            static void call(Component*& component, Managed* this_,           \
                HPX_ENUM_FWD_ARGS(N, T, t))                                   \
            {                                                                 \
                typedef typename Managed::wrapped_type wrapped_type;          \
                component = new wrapped_type(                                 \
                    HPX_ENUM_FORWARD_ARGS(N , T, t));                         \
                component->set_back_ptr(this_);                               \
            }                                                                 \
    /**/
            BOOST_PP_REPEAT_FROM_TO(1, HPX_COMPONENT_CREATE_ARGUMENT_LIMIT,
                MANAGED_COMPONENT_CONSTRUCT_INIT2, _)

#undef MANAGED_COMPONENT_CONSTRUCT_INIT2
        };

        ///////////////////////////////////////////////////////////////////////
        // This is used by the component implementation to decide whether to
        // delete the managed_component instance it depends on.
        template <typename DtorTag>
        struct destroy_backptr;

        template <>
        struct destroy_backptr<traits::managed_object_is_lifetime_controlled>
        {
            template <typename BackPtr>
            static void call(BackPtr* back_ptr)
            {
                // The managed_component's controls the lifetime of the
                // component implementation.
                delete back_ptr;
            }
        };

        template <>
        struct destroy_backptr<traits::managed_object_controls_lifetime>
        {
            template <typename BackPtr>
            static void call(BackPtr*)
            {
                // The managed_component's lifetime is controlled by the
                // component implementation. Do nothing.
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // This is used by the managed_component to decide whether to
        // delete the component implementation depending on it.
        template <typename DtorTag>
        struct manage_lifetime;

        template <>
        struct manage_lifetime<traits::managed_object_is_lifetime_controlled>
        {
            template <typename Component>
            static void call(Component*)
            {
                // The managed_component's lifetime is controlled by the
                // component implementation. Do nothing.
            }

            template <typename Component>
            static void addref(Component* component)
            {
                intrusive_ptr_add_ref(component);
            }

            template <typename Component>
            static void release(Component* component)
            {
                intrusive_ptr_release(component);
            }
        };

        template <>
        struct manage_lifetime<traits::managed_object_controls_lifetime>
        {
            template <typename Component>
            static void call(Component* component)
            {
                // The managed_component's controls the lifetime of the
                // component implementation.
                component->finalize();
                delete component;
            }

            template <typename Component>
            static void addref(Component*)
            {
            }

            template <typename Component>
            static void release(Component*)
            {
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Wrapper,
        typename CtorPolicy, typename DtorPolicy>
    class managed_component_base
      : public detail::managed_component_tag, boost::noncopyable
    {
    public:
        typedef typename boost::mpl::if_<
            boost::is_same<Component, detail::this_type>,
            managed_component_base, Component
        >::type this_component_type;

        typedef this_component_type wrapped_type;

        typedef void has_managed_component_base;
        typedef CtorPolicy ctor_policy;
        typedef DtorPolicy dtor_policy;

        // make sure that we have a back_ptr whenever we need to control the
        // lifetime of the managed_component
        BOOST_STATIC_ASSERT((
            boost::is_same<ctor_policy, traits::construct_without_back_ptr>::value ||
            boost::is_same<dtor_policy, traits::managed_object_controls_lifetime>::value));

        managed_component_base()
          : back_ptr_(0)
        {}

        managed_component_base(managed_component<Component, Wrapper>* back_ptr)
          : back_ptr_(back_ptr)
        {
            HPX_ASSERT(back_ptr);
        }

        // The implementation of the component is responsible for deleting the
        // actual managed component object
        ~managed_component_base()
        {
            detail_adl_barrier::destroy_backptr<dtor_policy>::call(back_ptr_);
        }

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

        naming::id_type get_gid() const;

        naming::gid_type get_base_gid() const;

        /// This is the default hook implementation for decorate_action which
        /// does no hooking at all.
        static HPX_STD_FUNCTION<threads::thread_function_type>
        wrap_action(HPX_STD_FUNCTION<threads::thread_function_type> f,
            naming::address::address_type)
        {
            return boost::move(f);
        }
    private:
        template <typename>
        friend struct detail_adl_barrier::init;

        void set_back_ptr(components::managed_component<Component, Wrapper>* bp)
        {
            HPX_ASSERT(0 == back_ptr_);
            HPX_ASSERT(bp);
            back_ptr_ = bp;
        }

        managed_component<Component, Wrapper>* back_ptr_;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Component, typename Derived>
        struct heap_factory
        {
            // the memory for the wrappers is managed by a one_size_heap_list
            typedef detail::wrapper_heap_list<
                detail::fixed_wrapper_heap<Derived> > heap_type;

            typedef detail::fixed_wrapper_heap<Derived> block_type;
            typedef Derived value_type;

        private:
            struct wrapper_heap_tag {};

            static heap_type& get_heap()
            {
                // ensure thread-safe initialization
                util::reinitializable_static<
                    heap_type, wrapper_heap_tag
                > heap(components::get_component_type<Component>());
                return heap.get();
            }

        public:
            static void* alloc(std::size_t count = 1)
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
    }
    // reference counting
    template <typename Component, typename Derived>
    void intrusive_ptr_add_ref(managed_component<Component, Derived>* p)
    {
        detail_adl_barrier::manage_lifetime<
            typename traits::managed_component_dtor_policy<Component>::type
        >::addref(p->component_);
    }
    template <typename Component, typename Derived>
    void intrusive_ptr_release(managed_component<Component, Derived>* p)
    {
        detail_adl_barrier::manage_lifetime<
            typename traits::managed_component_dtor_policy<Component>::type
        >::release(p->component_);
    }

    ///////////////////////////////////////////////////////////////////////////
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
    public:
        typedef typename boost::mpl::if_<
                boost::is_same<Derived, detail::this_type>,
                managed_component, Derived
            >::type derived_type;

        typedef Component wrapped_type;
        typedef Component type_holder;
        typedef typename Component::base_type_holder base_type_holder;

        typedef detail::heap_factory<Component, derived_type> heap_type;
        typedef typename heap_type::value_type value_type;

        /// \brief Construct a managed_component instance holding a
        ///        wrapped instance. This constructor takes ownership of the
        ///        passed pointer.
        ///
        /// \param c    [in] The pointer to the wrapped instance. The
        ///             managed_component takes ownership of this pointer.
        explicit managed_component(Component* comp)
          : component_(comp)
        {
            detail_adl_barrier::init<
                typename traits::managed_component_ctor_policy<Component>::type
            >::call(component_, this);
            intrusive_ptr_add_ref(this);
        }

    public:
        /// \brief Construct a managed_component instance holding a new wrapped
        ///        instance
        managed_component()
          : component_(0)
        {
            detail_adl_barrier::init<
                typename traits::managed_component_ctor_policy<Component>::type
            >::call_new(component_, this);
            intrusive_ptr_add_ref(this);
        }

#define MANAGED_COMPONENT_CONSTRUCT(Z, N, _)                                  \
        template <BOOST_PP_ENUM_PARAMS(N, typename T)>                        \
        managed_component(HPX_ENUM_FWD_ARGS(N, T, t))                         \
          : component_(0)                                                     \
        {                                                                     \
            detail_adl_barrier::init<                                         \
                typename traits::managed_component_ctor_policy<Component>::type \
            >::call(component_, this, HPX_ENUM_FORWARD_ARGS(N, T, t));        \
            intrusive_ptr_add_ref(this);                                      \
        }                                                                     \
    /**/

        BOOST_PP_REPEAT_FROM_TO(1, HPX_COMPONENT_CREATE_ARGUMENT_LIMIT,
            MANAGED_COMPONENT_CONSTRUCT, _)

#undef MANAGED_COMPONENT_CONSTRUCT

    public:
        /// \brief The destructor releases any wrapped instances
        ~managed_component()
        {
            intrusive_ptr_release(this);
            detail_adl_barrier::manage_lifetime<
                typename traits::managed_component_dtor_policy<Component>::type
            >::call(component_);
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
            if (!component_) {
                hpx::util::osstream strm;
                strm << "component is NULL ("
                     << components::get_component_type_name(
                        components::get_component_type<wrapped_type>())
                     << ") gid(" << get_base_gid() << ")";
                HPX_THROW_EXCEPTION(invalid_status,
                    "managed_component<Component, Derived>::get_checked",
                    hpx::util::osstream_get_string(strm));
            }
            return get();
        }

        Component const* get_checked() const
        {
            if (!component_) {
                hpx::util::osstream strm;
                strm << "component is NULL ("
                     << components::get_component_type_name(
                        components::get_component_type<wrapped_type>())
                     << ") gid(" << get_base_gid() << ")";
                HPX_THROW_EXCEPTION(invalid_status,
                    "managed_component<Component, Derived>::get_checked",
                    hpx::util::osstream_get_string(strm));
            }
            return get();
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
            void* p = heap_type::alloc();
            if (NULL == p) {
                HPX_THROW_STD_EXCEPTION(std::bad_alloc(),
                    "managed_component::operator new(std::size_t size)");
            }
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
        static value_type* create(std::size_t count)
        {
            // allocate the memory
            void* p = heap_type::alloc(count);
            if (NULL == p) {
                HPX_THROW_STD_EXCEPTION(std::bad_alloc(),
                    "managed_component::create");
            }

            if (1 == count)
                return new (p) value_type();

            // call constructors
            std::size_t succeeded = 0;
            try {
                value_type* curr = reinterpret_cast<value_type*>(p);
                for (std::size_t i = 0; i != count; ++i, ++curr) {
                    // call placement new, might throw
                    new (curr) value_type();
                    ++succeeded;
                }
            }
            catch (...) {
                // call destructors for successfully constructed objects
                value_type* curr = reinterpret_cast<value_type*>(p);
                for (std::size_t i = 0; i != succeeded; ++i)
                {
                    curr->finalize();
                    curr->~derived_type();
                    ++curr;
                }
                heap_type::free(p, count);     // free memory
                throw;      // rethrow
            }
            return reinterpret_cast<value_type*>(p);
        }

        /// \brief  The function \a destroy is used for deletion and
        //          de-allocation of arrays of wrappers
        static void destroy(value_type* p, std::size_t count = 1)
        {
            if (NULL == p || 0 == count)
                return;     // do nothing if given a NULL pointer

            // call destructors for all managed_component instances
            value_type* curr = p;
            for (std::size_t i = 0; i != count; ++i)
            {
                curr->finalize();
                curr->~derived_type();
                ++curr;
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
            return factory_none;
        }

#if defined(HPX_HAVE_SECURITY)
        static components::security::capability get_required_capabilities(
            components::security::traits::capability<>::capabilities caps)
        {
            return components::default_component_creation_capabilities(caps);
        }
#endif

    public:

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
        /// \brief Return the global id of this \a future instance
        naming::id_type get_gid() const
        {
            return naming::id_type(get_base_gid(), naming::id_type::unmanaged);
        }
        naming::gid_type get_base_gid() const
        {
            return heap_type::get_gid(const_cast<managed_component*>(this));
        }

    public:
        // reference counting
        template<typename C, typename D>
        friend void intrusive_ptr_add_ref(managed_component<C, D>* p);

        template<typename C, typename D>
        friend void intrusive_ptr_release(managed_component<C, D>* p);

    protected:
        Component* component_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Component, typename Wrapper,
        typename CtorPolicy, typename DtorPolicy>
    inline naming::id_type
    managed_component_base<Component, Wrapper, CtorPolicy, DtorPolicy>::get_gid() const
    {
        HPX_ASSERT(back_ptr_);
        return back_ptr_->get_gid();
    }

    template <typename Component, typename Wrapper,
        typename CtorPolicy, typename DtorPolicy>
    inline naming::gid_type
    managed_component_base<Component, Wrapper, CtorPolicy, DtorPolicy>::get_base_gid() const
    {
        HPX_ASSERT(back_ptr_);
        return back_ptr_->get_base_gid();
    }
}}

#endif


