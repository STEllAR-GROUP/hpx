//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_FULLEMPTYMEMORY_JUN_16_2008_1102AM)
#define HPX_UTIL_FULLEMPTYMEMORY_JUN_16_2008_1102AM

#include <boost/noncopyable.hpp>
#include <boost/call_traits.hpp>
#include <boost/aligned_storage.hpp>
#include <boost/type_traits/alignment_of.hpp>
#include <boost/type_traits/add_pointer.hpp>

#include <hpx/util/static.hpp>
#include <hpx/util/full_empty_store.hpp>

namespace hpx { namespace util
{
    namespace detail
    {
        class full_empty_base : boost::noncopyable
        {
        protected:
            // there is exactly one empty/full store where all memory blocks are 
            // stored
            typedef detail::full_empty_store store_type;
            struct full_empty_tag {};
            
            static store_type& get_store()
            {
                // ensure thread-safe initialization
                static util::static_<store_type, full_empty_tag> store;
                return store.get();
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \class full_empty full_empty_memory.hpp hpx/lcos/full_empty_memory.hpp
    ///
    /// The \a full_empty data type is a implementation of memory areas guarded
    /// by full/empty bits, a very low level synchronization primitive. The 
    /// class has been modeled after the FEB implementation in the QThread
    /// library (see here: http://www.cs.sandia.gov/qthreads/).
    ///
    /// All member functions but \a set_empty and \a set_full have the potential 
    /// of blocking until the corresponding precondition is met. Memory is 
    /// assumed to be full unless otherwise asserted, and as such memory that 
    /// is full and does not have dependencies (i.e. no threads are waiting for 
    /// it to become empty) does not require state data to be stored. It is 
    /// expected that while there may be locks instantiated at one time or 
    /// another for a very large number of addresses in the system, relatively 
    /// few will be in a non-default (full, no waiters) state at any one time.
    ///
    /// The main idea is, that a memory location can be either empty or full.
    /// setting or writing a value to a location sets it to full. Reading from 
    /// a location retrieves the value and (optionally) sets it to empty. A 
    /// write will block if the location is full and will wait for it to become
    /// empty. If several write's are waiting for a location to become empty 
    /// only one thread will be re-activated the moment it gets empty. A read 
    /// will block if the location is empty and will wait for it to become
    /// full. If several read's are waiting for a location to become full 
    /// all threads will be re-activated the moment it gets full. 
    /// 
    /// full_empty memory locations are very useful for synchronization and 
    /// data delivery (especially in producer/consumer scenarios).
    ///
    /// \tparam T   The template parameter \a T defines the type of the memory 
    ///             location to be guarded by an empty/full bit. It is possible 
    ///             to use any C++ data type with the empty/full mechanism.
    ///             If you want to use the empty/full synchronization facilities
    ///             without having to transfer (read/write) any data you can
    ///             use the specialization lcos#full_empty<void>.
    template <typename T>
    class full_empty : public detail::full_empty_base
    {
    private:
        typedef T value_type;

    public:
        /// \brief Create a new full/empty storage in empty state
        full_empty() 
        {
            ::new (get_address()) T();    // properly initialize memory
            set_empty();
        }

        /// \brief Destruct the full/empty data item
        ~full_empty()
        {
            BOOST_ASSERT(!get_store().is_used(get_address()));
            get_store().remove(get_address());
            get_address()->T::~T();       // properly destruct value in memory
        }

        /// \brief Atomically set the state to empty without releasing any 
        ///        waiting \a threads. This function is mainly usable for
        ///        initialization and debugging purposes.
        /// 
        /// \note    This function will create a new full/empty entry in the 
        ///          store if it doesn't exist yet.
        void set_empty()
        {
            get_store().set_empty(get_address());
        }

        /// \brief Atomically set the state to full without releasing any 
        ///        waiting \a threads. This function is mainly usable for
        ///        initialization and debugging purposes.
        /// 
        /// \note    This function will not create a new full/empty entry in 
        ///          the store if it doesn't exist yet.
        void set_full()
        {
            get_store().set_full(get_address());
        }

        /// \brief Query the current state of the memory
        bool is_empty() const
        {
            return get_store().is_empty(get_address());
        }

        /// \brief  Waits for the memory to become full and then reads it, 
        ///         leaves memory in full state. If the location is empty the 
        ///         calling thread will wait (block) for another thread to call 
        ///         either the function \a set or the function \a write.
        ///
        /// \note   When memory becomes full, all \a threads waiting for it
        ///         to become full with a read will receive the value at once 
        ///         and will be queued to run.
        void read(threads::thread_self& self, value_type& dest)
        {
            get_store().read(self, get_address(), dest);
        }

        /// \brief  Waits for memory to become full and then reads it, sets 
        ///         memory to empty. If the location is empty the calling 
        ///         thread will wait (block) for another thread to call either
        ///         the function \a set or the function \a write.
        ///
        /// \note   When memory becomes empty, only one thread blocked like this 
        ///         will be queued to run (one thread waiting in a \a write 
        ///         function).
        void read_and_empty(threads::thread_self& self, value_type& dest) 
        {
            get_store().read_and_empty(self, get_address(), dest);
        }

        /// \brief  Writes memory and atomically sets its state to full without 
        ///         waiting for it to become empty.
        /// 
        /// \note   Even if the function itself doesn't block, setting the 
        ///         location to full using \a set might re-activate threads 
        ///         waiting on this in a \a read or \a read_and_empty function.
        void set(value_type const& data)
        {
            get_store().set(get_address(), data);
        }

        /// \brief  Waits for memory to become empty, and then fills it. If the 
        ///         location is filled the calling thread will wait (block) for 
        ///         another thread to call the function \a read_and_empty.
        ///
        /// \note   When memory becomes empty only one thread blocked like this 
        ///         will be queued to run.
        void write(threads::thread_self& self, value_type const& data)
        {
            get_store().write(self, get_address(), data);
        }

    private:
        // type safe accessors to the stored data
        typedef typename boost::add_pointer<value_type>::type pointer;
        typedef typename boost::add_pointer<value_type const>::type const_pointer;

        pointer get_address()
        {
            return static_cast<pointer>(data_.address());
        }
        const_pointer get_address() const
        {
            return static_cast<const_pointer>(data_.address());
        }

        // the stored data needs to be properly aligned
        typedef boost::aligned_storage<sizeof(value_type),
            boost::alignment_of<value_type>::value> storage_type;

        storage_type data_;
    };

    /// \class full_empty full_empty_memory.hpp hpx/lcos/full_empty_memory.hpp
    /// The full_empty<void> is a specialization of the lcos#full_empty 
    /// template which is useful mainly as a synchronization primitive.
    template <>
    class full_empty<void> : public detail::full_empty_base
    {
    public:
        /// \brief Create a new full/empty storage in empty state
        full_empty() 
        {
            set_empty();
        }

        /// \brief Destruct the full/empty data item
        ~full_empty()
        {
            BOOST_ASSERT(!get_store().is_used(get_address()));
            get_store().remove(get_address());
        }

        /// \brief Atomically set the state to empty without releasing any 
        ///        waiting \a threads. This function is mainly usable for
        ///        initialization and debugging purposes.
        /// 
        /// \note    This function will create a new full/empty entry in the 
        ///          store if it doesn't exist yet.
        void set_empty()
        {
            get_store().set_empty(get_address());
        }

        /// \brief Atomically set the state to full without releasing any 
        ///        waiting \a threads. This function is mainly usable for
        ///        initialization and debugging purposes.
        /// 
        /// \note    This function will not create a new full/empty entry in 
        ///          the store if it doesn't exist yet.
        void set_full()
        {
            get_store().set_full(get_address());
        }

        /// \brief Query the current state of the memory
        bool is_empty() const
        {
            return get_store().is_empty(get_address());
        }

        /// Wait for the memory to become full, leaves memory in full state.
        ///
        /// \note When memory becomes full, all \a threads waiting for it
        ///       to become full with a read will be queued to run.
        void read(threads::thread_self& self)
        {
            get_store().read(self, get_address());
        }

        /// Wait for memory to become full, sets memory to empty.
        ///
        /// \note When memory becomes empty, only one thread blocked like this 
        ///       will be queued to run (one thread waiting in a \a write 
        ///       function).
        void read_and_empty(threads::thread_self& self) 
        {
            return get_store().read_and_empty(self, get_address());
        }

        /// \brief Writes memory and atomically sets its state to full without 
        ///        waiting for it to become empty.
        /// 
        /// \note  Even if the function itself doesn't block, setting the 
        ///        location to full using \a set might re-activate threads 
        ///        waiting on this in a \a read or \a read_and_empty function.
        void set()
        {
            get_store().set(get_address());
        }

        /// Wait for memory to become empty, and then fill it.
        ///
        /// \note When memory becomes empty only one thread blocked like this 
        ///       will be queued to run.
        void write(threads::thread_self& self)
        {
            get_store().write(self, get_address());
        }

    private:
        void* get_address() { return this; }
        void const* get_address() const { return this; }
    };

}}

#endif


