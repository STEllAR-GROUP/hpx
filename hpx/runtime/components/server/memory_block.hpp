//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_MEMORY_BLOCK_OCT_21_2008_0159PM)
#define HPX_COMPONENTS_MEMORY_BLOCK_OCT_21_2008_0159PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/server/wrapper_heap.hpp>
#include <hpx/runtime/components/server/wrapper_heap_list.hpp>
#include <hpx/runtime/actions/action.hpp>
#include <hpx/util/static.hpp>
#include <hpx/util/util.hpp>

#include <boost/noncopyable.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/detail/atomic_count.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server 
{
    class memory_block;     // forward declaration only

}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    /// \brief The memory_block_header holds all information needed to describe
    ///        a block of memory managed by a server#memory_block component.
    class memory_block_header : boost::noncopyable
    {
    public:
        /// This constructor is called on the locality where there memory_block
        /// is hosted
        memory_block_header(server::memory_block* wrapper, std::size_t size)
          : count_(0), size_(size), wrapper_(wrapper)
        {}

        /// This constructor is called whenever a memory_block gets 
        /// de-serialized
        explicit memory_block_header(std::size_t size)
          : count_(0), size_(size), wrapper_(NULL)
        {}

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

        /// Return whether this instance is the master instance of this
        /// memory block
        bool is_master() const { return 0 != wrapper_; }

        // This is the component id. Every component needs to have an embedded
        // enumerator 'value' which is used by the generic action implementation
        // to associate this component with a given action.
        static component_type get_component_type()
        {
            return component_memory_block;
        }
        static void set_component_type(component_type t)
        {
            BOOST_ASSERT(false);    // this shouldn't be ever called
        }

    protected:
        friend void intrusive_ptr_add_ref(memory_block_header* p);
        friend void intrusive_ptr_release(memory_block_header* p);

        boost::detail::atomic_count count_;
        std::size_t size_;
        server::memory_block* wrapper_;
    };

    /// support functions for boost::intrusive_ptr
    inline void intrusive_ptr_add_ref(memory_block_header* p)
    {
        ++p->count_;
    }

    inline void intrusive_ptr_release(memory_block_header* p)
    {
        if (0 == --p->count_)
            delete (boost::uint8_t*)p;    // memory was allocated as array of uint8_t's
    }

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
    public:
        memory_block_data()
        {}
        memory_block_data(boost::intrusive_ptr<server::detail::memory_block_header> data)
          : data_(data)
        {}

        /// \brief Return a pointer to the wrapped memory_block_data instance
        boost::uint8_t* get()
        {
            if (!data_) {
                HPX_OSSTREAM strm;
                strm << "memory_block_data data is NULL (" 
                     << components::get_component_type_name(component_memory_block) 
                     << ")";
                HPX_THROW_EXCEPTION(invalid_status, HPX_OSSTREAM_GETSTRING(strm));
            }
            return data_->get_ptr();
        }
        boost::uint8_t const* get() const
        {
            if (!data_) {
                HPX_OSSTREAM strm;
                strm << "memory_block_data data is NULL (" 
                     << components::get_component_type_name(component_memory_block)
                     << ")";
                HPX_THROW_EXCEPTION(invalid_status, HPX_OSSTREAM_GETSTRING(strm));
            }
            return data_->get_ptr();
        }

    private:
        /// A memory_block_data structure is just a wrapper for a block of 
        /// memory which has to be serialized as is
        ///
        /// \note Serializing memory_blocks is not platform independent as
        ///       such things as endianess, alignment, and data type sizes are
        ///       not considered.
        friend class boost::serialization::access;

        template<class Archive>
        void save(Archive & ar, const unsigned int version) const
        {
            std::size_t size = data_->get_size();
            ar << size;
            ar << boost::serialization::make_array(data_->get_ptr(), data_->get_size());
        }

        template<class Archive>
        void load(Archive & ar, const unsigned int version)
        {
            std::size_t size = 0;
            ar >> size;

            boost::uint8_t* p = new boost::uint8_t[
                size + sizeof(server::detail::memory_block_header)];
            new ((server::detail::memory_block_header*)p) 
                server::detail::memory_block_header(size);

            data_.reset((server::detail::memory_block_header*)p);

            ar >> boost::serialization::make_array(data_->get_ptr(), size);
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()

    private:
        boost::intrusive_ptr<server::detail::memory_block_header> data_;
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
    class memory_block : public memory_block_header
    {
    public:
        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            memory_block_get = 0,
            memory_block_checkout = 1,
            memory_block_checkin = 2,
        };

        memory_block(server::memory_block* wrapper, std::size_t size)
          : memory_block_header(wrapper, size)
        {}
        ~memory_block()
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
        threads::thread_state 
        get (threads::thread_self&, applier::applier& appl, 
            memory_block_data* result);

        components::memory_block_data local_get (applier::applier& appl);

        /// Get the current data for reading
        threads::thread_state 
        checkout (threads::thread_self& self, applier::applier& appl, 
            components::memory_block_data* result);

        components::memory_block_data local_checkout (applier::applier& appl);

        /// Write back data
        threads::thread_state 
        checkin (threads::thread_self&, applier::applier& appl, 
            components::memory_block_data const& newdata);

        void local_checkin (applier::applier& appl, 
            components::memory_block_data const& data);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::direct_result_action0<
            memory_block, components::memory_block_data, memory_block_get, 
            &memory_block::get, &memory_block::local_get
        > get_action;

        typedef hpx::actions::direct_result_action0<
            memory_block, components::memory_block_data, memory_block_checkout, 
            &memory_block::checkout, &memory_block::local_checkout
        > checkout_action;

        typedef hpx::actions::direct_action1<
            memory_block, memory_block_checkin, 
            components::memory_block_data const&, 
            &memory_block::checkin, &memory_block::local_checkin
        > checkin_action;
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

        /// \brief Construct an empty managed_component
        memory_block() 
          : component_(0) 
        {}

        /// \brief Construct a memory_block instance holding a memory_block 
        ///        instance. This constructor takes ownership of the 
        ///        passed pointer.
        ///
        /// \param c    [in] The pointer to the memory_block instance. The 
        ///             memory_block instance takes ownership of this pointer.
        explicit memory_block(wrapped_type* c) 
          : component_(c) 
        {}

        /// \brief Construct a managed_component instance holding a new wrapped
        ///        instance
        ///
        /// \param appl [in] The applier to be used for construction of the new
        ///             wrapped instance. 
        memory_block(applier::applier& appl, std::size_t size) 
          : component_(0) 
        {
            boost::uint8_t* p = new boost::uint8_t[size + sizeof(detail::memory_block_header)];
            new ((detail::memory_block*)p) detail::memory_block(this, size);
            component_.reset((detail::memory_block_header*)p);
        }

        /// \brief The destructor releases any wrapped instances
        ~memory_block()
        {}

        /// \brief Return a pointer to the wrapped memory_block instance
        detail::memory_block* get()
        {
            if (!component_) {
                HPX_OSSTREAM strm;
                strm << "component is NULL (" 
                     << components::get_component_type_name(component_memory_block) 
                     << ")";
                HPX_THROW_EXCEPTION(invalid_status, HPX_OSSTREAM_GETSTRING(strm));
            }
            return static_cast<detail::memory_block*>(component_.get());
        }
        detail::memory_block const* get() const
        {
            if (!component_) {
                HPX_OSSTREAM strm;
                strm << "component is NULL (" 
                     << components::get_component_type_name(component_memory_block)
                     << ")";
                HPX_THROW_EXCEPTION(invalid_status, HPX_OSSTREAM_GETSTRING(strm));
            }
            return static_cast<detail::memory_block const*>(component_.get());
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

    protected:
        // the memory for the wrappers is managed by a one_size_heap_list
        typedef components::detail::wrapper_heap_list<
            components::detail::fixed_wrapper_heap<memory_block> > 
        heap_type;

        struct wrapper_heap_tag {};

        static heap_type& get_heap()
        {
            // ensure thread-safe initialization
            static util::static_<heap_type, wrapper_heap_tag> heap;
            return heap.get();
        }

    public:
        /// \brief  The memory for managed_component objects is managed by 
        ///         a class specific allocator. This allocator uses a one size 
        ///         heap implementation, ensuring fast memory allocation.
        ///         Additionally the heap registers the allocated  
        ///         managed_component instance with the DGAS service.
        ///
        /// \param size   [in] The parameter \a size is supplied by the 
        ///               compiler and contains the number of bytes to allocate.
        static void* operator new(std::size_t size)
        {
            if (size > sizeof(memory_block))
                return ::operator new(size);
            return get_heap().alloc();
        }
        static void operator delete(void* p, std::size_t size)
        {
            if (NULL == p) 
                return;     // do nothing if given a NULL pointer

            if (size != sizeof(memory_block)) {
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
        //          initialization of components. Here we abuse the count 
        ///         parameter normally used to specify the number of objects to 
        ///         be created. It is interpreted as the number of bytes to 
        ///         allocate for the new memory_block.
        static memory_block* 
        create(applier::applier& appl, std::size_t count)
        {
            // allocate the memory
            memory_block* p = get_heap().alloc();
            return new (p) memory_block(appl, count);
        }

        /// \brief  The function \a destroy is used for deletion and 
        //          de-allocation of arrays of wrappers
        static void destroy(memory_block* p, std::size_t count)
        {
            if (NULL == p || 0 == count) 
                return;     // do nothing if given a NULL pointer

            p->~memory_block();

            // free memory itself
            get_heap().free(p);
        }

        /// \brief  The function \a has_multi_instance_factory is used to 
        ///         determine, whether instances of the derived component can 
        ///         be created in blocks (i.e. more than one instance at once). 
        ///         This function is used by the \a distributing_factory to 
        ///         determine a correct allocation strategy
        static bool has_multi_instance_factory()
        {
            // this component can be allocated one at a time only, but the 
            // meaning of the count parameter passed to create is different.
            // In this case it specifies the number of bytes to allocate for a
            // new memory block.

            // This assertion is in place to avoid creating this component
            // using the distributed factory (currently the only place this
            // function gets invoked from).
            BOOST_ASSERT(false);
            return true;
        }

    public:
        ///
        naming::id_type 
        get_gid(applier::applier& appl) const
        {
            return get_heap().get_gid(appl, const_cast<memory_block*>(this));
        }

    private:
        friend class detail::memory_block;

        /// \brief We use a intrusive pointer here to make sure the size of the
        ///        overall memory_block class is exactly equal to the size of
        ///        a single pointer
        boost::intrusive_ptr<detail::memory_block_header> component_;
    };

}}}

#endif
