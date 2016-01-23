//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_MEMORY_BLOCK_OCT_21_2008_0159PM)
#define HPX_COMPONENTS_MEMORY_BLOCK_OCT_21_2008_0159PM

#include <hpx/config.hpp>
#include <hpx/traits/is_component.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/pinned_ptr.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/wrapper_heap.hpp>
#include <hpx/runtime/components/server/wrapper_heap_list.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/actions/manage_object_action.hpp>
#include <hpx/runtime/serialization/detail/raw_ptr.hpp>
#include <hpx/lcos/base_lco_with_value.hpp>
#include <hpx/util/reinitializable_static.hpp>

#include <boost/noncopyable.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/detail/atomic_count.hpp>

#include <sstream>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    class memory_block;     // forward declaration only
    class runtime_support;
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    /// \brief The memory_block_header holds all information needed to describe
    ///        a block of memory managed by a server#memory_block component.
    class memory_block_header : boost::noncopyable //-V690
    {
    public:
        /// This constructor is called on the locality where there memory_block
        /// is hosted
        memory_block_header(server::memory_block* wrapper, std::size_t size,
                hpx::actions::manage_object_action_base const& act)
          : count_(0), size_(size), wrapper_(wrapper),
            managing_object_(act.get_instance())
        {
            HPX_ASSERT(act.construct());
            act.construct()(this->get_ptr(), size);
        }

        memory_block_header(server::memory_block* wrapper,
                memory_block_header const* rhs, std::size_t size,
                hpx::actions::manage_object_action_base const& act)
          : count_(0), size_(size), wrapper_(wrapper),
            managing_object_(act.get_instance())
        {
            HPX_ASSERT(act.clone());
            act.clone()(this->get_ptr(), rhs->get_ptr(), size);
        }

        /// This constructor is called whenever a memory_block gets
        /// de-serialized
        explicit memory_block_header(std::size_t size,
                hpx::actions::manage_object_action_base const& act)
          : count_(0), size_(size), wrapper_(NULL),
            managing_object_(act.get_instance())
        {
            HPX_ASSERT(act.construct());
            act.construct()(this->get_ptr(), size);
        }

        ~memory_block_header()
        {
            // invoke destructor, if needed
            HPX_ASSERT(this->managing_object_.destruct());
            this->managing_object_.destruct()(this->get_ptr());
        }

        memory_block_header& operator= (memory_block_header const& rhs)
        {
            if (this != &rhs) {
                HPX_ASSERT(this->managing_object_.assign());
                this->managing_object_.assign()(
                    this->get_ptr(), rhs.get_ptr(), size_);
            }
            return *this;
        }

        /// \brief get_ptr returns the address of the first byte allocated for
        ///        this memory_block.
        boost::uint8_t* get_ptr()
        {
            return reinterpret_cast<boost::uint8_t*>(this + 1);
        }
        boost::uint8_t const* get_ptr() const
        {
            return reinterpret_cast<boost::uint8_t const*>(this + 1);
        }

        /// return the size of the memory block contained in this instance
        std::size_t get_size() const { return size_; }

        /// return the reference to the managing object
        actions::manage_object_action_base const& get_managing_object() const
        {
            return managing_object_;
        }

        /// Return whether this instance is the master instance of this
        /// memory block
        bool is_master() const { return 0 != wrapper_; }

        static component_type get_component_type()
        {
            return components::get_component_type<memory_block_header>();
        }
        static void set_component_type(component_type t)
        {
            components::set_component_type<memory_block_header>(t);
        }

        naming::id_type get_id() const;
        naming::id_type get_unmanaged_id() const;

#if defined(HPX_HAVE_COMPONENT_GET_GID_COMPATIBILITY)
        naming::id_type get_gid() const
        {
            return get_unmanaged_id();
        }
#endif

    protected:
        naming::gid_type get_base_gid() const;

    protected:
        friend void intrusive_ptr_add_ref(memory_block_header* p);
        friend void intrusive_ptr_release(memory_block_header* p);

        boost::detail::atomic_count count_;
        std::size_t size_;
        server::memory_block* wrapper_;
        actions::manage_object_action_base const& managing_object_;
    };

    /// support functions for boost::intrusive_ptr
    inline void intrusive_ptr_add_ref(memory_block_header* p)
    {
        ++p->count_;
    }

    inline void intrusive_ptr_release(memory_block_header* p)
    {
        if (0 == --p->count_) {
            p->~memory_block_header();
            ::free(p);
        }
    }

    template <typename T>
    inline T* allocate_block(std::size_t size)
    {
        return static_cast<T*>(::malloc(size + sizeof(detail::memory_block_header)));
    }

    ///////////////////////////////////////////////////////////////////////////
    // forward declaration
    class memory_block;
}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    /// \brief The \a memory_block_data structure is used for marshaling the
    ///        memory managed by a server#memory_block_data component to a
    ///        remote site.
    class memory_block_data
    {
        // server::detail::memory_block needs access to our data_ member
        friend class components::server::detail::memory_block;

    public:
        memory_block_data()
        {}
        memory_block_data(boost::intrusive_ptr<server::detail::memory_block_header> data)
          : data_(data)
        {}
        memory_block_data(boost::intrusive_ptr<server::detail::memory_block_header> data,
                boost::intrusive_ptr<server::detail::memory_block_header> config)
          : data_(data), config_(config)
        {}

        /// \brief Return a pointer to the wrapped memory_block_data instance
        boost::uint8_t* get_ptr()
        {
            if (!data_) {
                std::ostringstream strm;
                strm << "memory_block_data data is NULL ("
                     << components::get_component_type_name(component_memory_block)
                     << ")";
                HPX_THROW_EXCEPTION(invalid_status,
                    "memory_block_data::get_ptr",
                    strm.str());
            }
            return data_->get_ptr();
        }
        boost::uint8_t const* get_ptr() const
        {
            if (!data_) {
                std::ostringstream strm;
                strm << "memory_block_data data is NULL ("
                     << components::get_component_type_name(component_memory_block)
                     << ")";
                HPX_THROW_EXCEPTION(invalid_status,
                    "memory_block_data::get_ptr const",
                    strm.str());
            }
            return data_->get_ptr();
        }

        std::size_t get_size() const
        {
            if (!data_) {
                std::ostringstream strm;
                strm << "memory_block_data data is NULL ("
                     << components::get_component_type_name(component_memory_block)
                     << ")";
                HPX_THROW_EXCEPTION(invalid_status,
                    "memory_block_data::get_size",
                    strm.str());
            }
            return data_->get_size();
        }

        template <typename T>
        T& get()
        {
            return *reinterpret_cast<T*>(get_ptr());
        }

        template <typename T>
        T const& get() const
        {
            return *reinterpret_cast<T const*>(get_ptr());
        }

        template <typename T>
        void set (T const& val)
        {
            if (!data_) {
                std::ostringstream strm;
                strm << "memory_block_data data is NULL ("
                     << components::get_component_type_name(component_memory_block)
                     << ")";
                HPX_THROW_EXCEPTION(invalid_status,
                    "memory_block_data::set",
                    strm.str());
            }
            if (!data_->is_master())
            {
                std::ostringstream strm;
                strm << "memory_block_data data is not checked out ("
                     << components::get_component_type_name(component_memory_block)
                     << ")";
                HPX_THROW_EXCEPTION(invalid_status,
                    "memory_block_data::set",
                    strm.str());
            }
            *reinterpret_cast<T*>(data_->get_ptr()) = val;
        }

    private:
        /// A memory_block_data structure is just a wrapper for a block of
        /// memory which has to be serialized as is
        ///
        /// \note Serializing memory_blocks is not platform independent as
        ///       such things as endianess, alignment, and data type sizes are
        ///       not considered.
        friend class hpx::serialization::access;

        ///////////////////////////////////////////////////////////////////////
        template <class Archive>
        static void save_(Archive & ar, const unsigned int version,
            server::detail::memory_block_header* data,
            server::detail::memory_block_header* config = 0)
        {
            std::size_t size = data->get_size();
            actions::manage_object_action_base* act =
                const_cast<actions::manage_object_action_base*>(
                    &data->get_managing_object().get_instance());

            HPX_ASSERT(act);

            ar << size; //-V128
            ar << hpx::serialization::detail::raw_ptr(act);

            HPX_ASSERT(act->save());
            if (config) {
                act->save()(data->get_ptr(), data->get_size(), ar, version,
                    config->get_ptr());
            }
            else {
                act->save()(data->get_ptr(), data->get_size(), ar, version, 0);
            }
        }

        template <class Archive>
        void save(Archive & ar, const unsigned int version) const
        {
            bool has_config = config_ != 0;
            ar << has_config;
            if (has_config)
                save_(ar, version, config_.get());
            save_(ar, version, data_.get(), config_.get());
        }

        ///////////////////////////////////////////////////////////////////////
        template <class Archive>
        static server::detail::memory_block_header*
        load_(Archive & ar, const unsigned int version,
            server::detail::memory_block_header* config = 0)
        {
            std::size_t size = 0;
            actions::manage_object_action_base* act = 0;

            ar >> size; //-V128
            ar >> hpx::serialization::detail::raw_ptr(act);

            typedef server::detail::memory_block_header alloc_type;
            alloc_type* p =
                new (server::detail::allocate_block<alloc_type>(size))
                    alloc_type(size, act->get_instance()); //-V522

            HPX_ASSERT(act->load()); //-V522
            if (config) {
                act->load()(p->get_ptr(), size, ar, version, //-V522
                    config->get_ptr());
            }
            else {
                act->load()(p->get_ptr(), size, ar, version, 0); //-V522
            }

            delete act;

            return p;
        }

        template <class Archive>
        void load(Archive & ar, const unsigned int version)
        {
            bool has_config;
            ar >> has_config;
            if (has_config)
                config_.reset(load_(ar, version));
            data_.reset(load_(ar, version, config_.get()));
        }
        HPX_SERIALIZATION_SPLIT_MEMBER();

    private:
        boost::intrusive_ptr<server::detail::memory_block_header> data_;
        boost::intrusive_ptr<server::detail::memory_block_header> config_;
    };
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    /// \brief The memory_block structure implements the component
    ///        functionality exposed by a memory_block component
    ///
    /// The memory block this structure it is managing is constructed from a
    /// memory_block_header directly followed by the actual raw memory.
    class HPX_EXPORT memory_block : public memory_block_header
    {
    public:
        // construct a new memory block
        memory_block(server::memory_block* wrapper, std::size_t size,
                hpx::actions::manage_object_action_base const& act)
          : memory_block_header(wrapper, size, act)
        {}

        // clone memory from rhs
        memory_block(server::memory_block* wrapper, std::size_t size,
                detail::memory_block_header const* rhs,
                hpx::actions::manage_object_action_base const& act)
          : memory_block_header(wrapper, rhs, size, act)
        {}

        ///////////////////////////////////////////////////////////////////////
        // components must contain a typedef for wrapping_type defining the
        // managed_component type used to encapsulate instances of this
        // component
        typedef server::memory_block wrapping_type;
        typedef memory_block_header wrapped_type;

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Get the current data for reading
        components::memory_block_data get();

        /// Get the current data for reading, use config data for serialization
        components::memory_block_data get_config(
            components::memory_block_data const& cfg);

        /// Get the current data for reading
        components::memory_block_data checkout();

        /// Write back data
        void checkin(components::memory_block_data const& data);

        /// Clone this memory_block
        naming::gid_type clone();

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        HPX_DEFINE_COMPONENT_ACTION(memory_block, get);
        HPX_DEFINE_COMPONENT_ACTION(memory_block, get_config);
        HPX_DEFINE_COMPONENT_ACTION(memory_block, checkout);
        HPX_DEFINE_COMPONENT_ACTION(memory_block, checkin);
        HPX_DEFINE_COMPONENT_ACTION(memory_block, clone);

        // This component type requires valid id for its actions to be invoked
        static bool is_target_valid(naming::id_type const& id) { return true; }

        /// This is the default hook implementation for decorate_action which
        /// does no hooking at all.
        template <typename F>
        static threads::thread_function_type
        decorate_action(naming::address::address_type, F && f)
        {
            return std::forward<F>(f);
        }

        /// This is the default hook implementation for schedule_thread which
        /// forwards to the default scheduler.
        static void schedule_thread(naming::address::address_type,
            threads::thread_init_data& data,
            threads::thread_state_enum initial_state)
        {
            hpx::threads::register_work_plain(data, initial_state); //-V106
        }
    };
}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    ///
    class memory_block : boost::noncopyable
    {
    public:
        typedef detail::memory_block_header wrapped_type;
        typedef memory_block type_holder;

        /// \brief Construct an empty managed_component
        memory_block()
          : component_(0)
        {}

    private:
        /// \brief Construct a memory_block instance holding a memory_block
        ///        instance. This constructor takes ownership of the
        ///        passed pointer.
        ///
        /// \param c    [in] The pointer to the memory_block instance. The
        ///             memory_block instance takes ownership of this pointer.
        explicit memory_block(wrapped_type* c)
          : component_(c)
        {}

        /// \brief Construct a memory_block instance holding a new wrapped
        ///        instance
        ///
        /// \param appl [in] The applier to be used for construction of the new
        ///             wrapped instance.
        memory_block(std::size_t size,
                actions::manage_object_action_base const& act)
          : component_(0)
        {
            typedef detail::memory_block alloc_type;
            alloc_type* p = server::detail::allocate_block<alloc_type>(size);
            new (p) alloc_type(this, size, act);
            component_.reset(p);
        }

        /// \brief Construct a memory_block instance as a plain copy of the
        ///        parameter
        memory_block(detail::memory_block_header const* rhs,
                actions::manage_object_action_base const& act)
          : component_(0)
        {
            std::size_t size = rhs->get_size();
            typedef detail::memory_block alloc_type;
            alloc_type* p = server::detail::allocate_block<alloc_type>(size);
            new (p) alloc_type(this, size, rhs, act);
            component_.reset(p);
        }

        /// \brief finalize() will be called just before the instance gets
        ///        destructed
        ///
        /// \param self [in] The HPX \a thread used to execute this function.
        /// \param appl [in] The applier to be used for finalization of the
        ///             component instance.
        void finalize() {}

    public:
        /// \brief The destructor releases any wrapped instances
        ~memory_block()
        {}

        /// \brief Return a pointer to the wrapped memory_block instance
        detail::memory_block* get()
        {
            if (!component_) {
                std::ostringstream strm;
                strm << "component is NULL ("
                     << components::get_component_type_name(component_memory_block)
                     << ")";
                HPX_THROW_EXCEPTION(invalid_status,
                    "memory_block::get",
                    strm.str());
            }
            return static_cast<detail::memory_block*>(component_.get());
        }
        detail::memory_block const* get() const
        {
            if (!component_) {
                std::ostringstream strm;
                strm << "component is NULL ("
                     << components::get_component_type_name(component_memory_block)
                     << ")";
                HPX_THROW_EXCEPTION(invalid_status,
                    "memory_block::get const",
                    strm.str());
            }
            return static_cast<detail::memory_block const*>(component_.get());
        }

        detail::memory_block* get_checked()
        {
            return get();
        }
        detail::memory_block const* get_checked() const
        {
            return get();
        }

        static component_type get_component_type()
        {
            return wrapped_type::get_component_type();
        }
        static void set_component_type(component_type t)
        {
            wrapped_type::set_component_type(t);
        }

    protected:
        // the memory for the wrappers is managed by a one_size_heap_list
        typedef components::detail::wrapper_heap_list<
            components::detail::fixed_wrapper_heap<memory_block> >
        heap_type;

        struct wrapper_heap_tag {};

        static heap_type& get_heap()
        {
            // ensure thread-safe initialization
            util::reinitializable_static<
                heap_type, wrapper_heap_tag, HPX_RUNTIME_INSTANCE_LIMIT
            > heap(component_memory_block);
            return heap.get(get_runtime_instance_number());
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
            if (size > sizeof(memory_block))
                return ::malloc(size);
            return get_heap().alloc();
        }
        static void operator delete(void* p, std::size_t size)
        {
            if (NULL == p)
                return;     // do nothing if given a NULL pointer

            if (size != sizeof(memory_block)) {
                ::free(p);
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
        //          initialization of components. Here we abuse the count
        ///         parameter normally used to specify the number of objects to
        ///         be created. It is interpreted as the number of bytes to
        ///         allocate for the new memory_block.
        static memory_block* create(std::size_t count,
            actions::manage_object_action_base const& act)
        {
            // allocate the memory
            void* p = get_heap().alloc();
            return new (p) memory_block(count, act);
        }

        static memory_block* create(detail::memory_block_header const* rhs,
            actions::manage_object_action_base const& act)
        {
            // allocate the memory
            void* p = get_heap().alloc();
            return new (p) memory_block(rhs, act);
        }

        /// \brief  The function \a destroy is used for deletion and
        //          de-allocation of arrays of wrappers
        static void destroy(memory_block* p, std::size_t count = 1)
        {
            if (NULL == p || 0 == count)
                return;     // do nothing if given a NULL pointer

            p->finalize();
            p->~memory_block();

            // free memory itself
            get_heap().free(p);
        }

    public:
        /// \brief Return the global id of this \a future instance
        naming::id_type get_id() const
        {
            return get_checked()->get_id();
        }

        naming::id_type get_unmanaged_id() const
        {
            return get_checked()->get_unmanaged_id();
        }

#if defined(HPX_HAVE_COMPONENT_GET_GID_COMPATIBILITY)
        naming::id_type get_gid() const
        {
            return get_checked()->get_gid();
        }
#endif

    protected:
        friend class server::detail::memory_block;
        friend class server::detail::memory_block_header;
        friend class server::runtime_support;

        naming::gid_type get_base_gid() const
        {
            return get_heap().get_gid(const_cast<memory_block*>(this));
        }

    private:
        /// \brief We use a intrusive pointer here to make sure the size of the
        ///        overall memory_block class is exactly equal to the size of
        ///        a single pointer
        boost::intrusive_ptr<detail::memory_block_header> component_;
    };

    namespace detail
    {
        inline naming::id_type memory_block_header::get_id() const
        {
            return naming::id_type(get_base_gid(), naming::id_type::unmanaged);
        }

        inline naming::id_type memory_block_header::get_unmanaged_id() const //-V524
        {
            return naming::id_type(get_base_gid(), naming::id_type::unmanaged);
        }

        inline naming::gid_type memory_block_header::get_base_gid() const
        {
            HPX_ASSERT(wrapper_);
            return wrapper_->get_base_gid();
        }
    }
}}}

namespace hpx { namespace traits
{
    // memory_block is a (hand-rolled) component
    template <>
    struct is_component<components::server::memory_block>
      : boost::mpl::true_
    {};
}}

///////////////////////////////////////////////////////////////////////////////
// Declaration of serialization support for the memory_block actions
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::detail::memory_block::get_action,
    memory_block_get_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::detail::memory_block::get_config_action,
    memory_block_get_config_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::detail::memory_block::checkout_action,
    memory_block_checkout_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::detail::memory_block::checkin_action,
    memory_block_checkin_action)
HPX_REGISTER_ACTION_DECLARATION(
    hpx::components::server::detail::memory_block::clone_action,
    memory_block_clone_action)

HPX_REGISTER_BASE_LCO_WITH_VALUE_DECLARATION(
    hpx::components::memory_block_data, hpx_memory_data_type)

#include <hpx/config/warnings_suffix.hpp>

#endif
