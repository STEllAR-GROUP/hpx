//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_MEMORY_BLOCK_OCT_21_2008_0159PM)
#define HPX_COMPONENTS_MEMORY_BLOCK_OCT_21_2008_0159PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/actions/action.hpp>

#include <boost/noncopyable.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>

namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    /// Data structure used to access the memory encapsulated by a 
    /// memory_block component.
    class memory_block_data 
    {
    public:
        memory_block_data()
          : size_(0), is_readonly_(false)
        {
        }
        explicit memory_block_data(std::size_t size)
          : data_(new boost::uint8_t[size]), size_(size), is_readonly_(false)
        {
        }
        ~memory_block_data()
        {
        }

        template <typename T>
        T* get() 
        {
            if (is_readonly_) {
                HPX_THROW_EXCEPTION(invalid_status, 
                    "The memory_block is readonly");
                return NULL;
            }
            if (sizeof(T) < size_) {
                HPX_THROW_EXCEPTION(invalid_status, 
                    "The memory_block is too small for the requested data type");
                return NULL;
            }
            return reinterpret_cast<T*>(data_.get());
        }

        template <typename T>
        T const* get() const
        {
            if (sizeof(T) < size_) {
                HPX_THROW_EXCEPTION(invalid_status, 
                    "The memory_block is too small for the requested data type");
                return NULL;
            }
            return reinterpret_cast<T const*>(data_.get());
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
            ar << size_;
            ar << boost::serialization::make_array(data_.get(), size_);
        }

        template<class Archive>
        void load(Archive & ar, const unsigned int version)
        {
            ar >> size_;
            boost::uint8_t* data = new boost::uint8_t[size_];
            ar >> boost::serialization::make_array(data, size_);
            data_.reset(data);
            is_readonly_ = true;    // any received memory_block is readonly
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()

    private:
        boost::shared_ptr<boost::uint8_t> data_;
        std::size_t size_;
        bool is_readonly_;
    };

}}

namespace hpx { namespace components { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    /// \class memory_block memory_block.hpp hpx/runtime/components/server/memory_block.hpp
    ///
    /// 
    class memory_block : boost::noncopyable
    {
    public:
        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            memory_block_get = 0,
        };

        explicit memory_block(applier::applier&, std::size_t size = 0)
          : data_(size)
        {}

        ///////////////////////////////////////////////////////////////////////
        // components must contain a typedef for wrapping_type defining the
        // managed_component type used to encapsulate instances of this 
        // component
        typedef memory_block wrapping_type;
        typedef memory_block wrapped_type;

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

        /// \brief  The function \a create is used for allocation and 
        //          initialization of components. Here we abuse the count 
        ///         parameter normally used to specify the number of objects to 
        ///         be created. It is interpreted as the number of bytes to 
        ///         allocate for the new memory_block.
        static memory_block* 
        create(applier::applier& appl, std::size_t count)
        {
            return new memory_block(appl, count);
        }

        /// \brief  The function \a destroy is used for deletion and 
        //          de-allocation of components.
        static void destroy(memory_block* p, std::size_t count)
        {
            delete p;
        }

    public:
        /// Return the gid of this memory_block. By convention the gid is built
        /// from the locality prefix and the local virtual address of the 
        /// this memory_block instance
        naming::id_type 
        get_gid(applier::applier& appl) const
        {
            naming::id_type gid(appl.get_prefix()); 
            gid.set_lsb(reinterpret_cast<boost::uint64_t>(this));
            return gid;
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Initialize the accumulator
        threads::thread_state 
        get (threads::thread_self&, applier::applier& appl, 
            memory_block_data* result) 
        {
            *result = data_;
            return threads::terminated;
        }

        memory_block_data local_get (applier::applier& appl) 
        {
            return data_;
        }

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::direct_result_action0<
            memory_block, memory_block_data, memory_block_get, 
            &memory_block::get, &memory_block::local_get
        > get_action;

    private:
        memory_block_data data_;
    };

}}}

#endif
