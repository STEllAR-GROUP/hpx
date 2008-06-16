//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_FULLEMPTYMEMORY_JUN_16_2008_1102AM)
#define HPX_LCOS_FULLEMPTYMEMORY_JUN_16_2008_1102AM

#include <boost/noncopyable.hpp>
#include <boost/call_traits.hpp>
#include <boost/aligned_storage.hpp>
#include <boost/type_traits/alignment_of.hpp>
#include <boost/type_traits/add_pointer.hpp>

#include <hpx/lcos/full_empty_store.hpp>

namespace hpx { namespace lcos
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a full_empty data type is a implementation of memory areas guarded
    /// by full/empty bits, a very low level synchronization primitive. The 
    /// class has been modeled after the FEB implementation in the QThread
    /// library (see here: http://www.cs.sandia.gov/qthreads/).
    ///
    /// All member functions but \a empty and \a fill have the potential of 
    /// blocking until the corresponding precondition is met. Memory is assumed
    /// to be full unless otherwise asserted, and as such memory that is full 
    /// and does not have dependencies (i.e. no threads are waiting for it to 
    /// become empty) does not require state data to be stored. It is expected 
    /// that while there may be locks instantiated at one time or another for a 
    /// very large number of addresses in the system, relatively few will be in 
    /// a non-default (full, no waiters) state at any one time.
    ///
    /// \note This class is mostly not implemented yet
    ///
    template <typename T>
    class full_empty : boost::noncopyable
    {
    private:
        typedef T value_type;
        
    public:
        /// Create a new full/empty storage in empty state
        full_empty(threadmanager::px_thread_self& self) 
        {
            empty(self);
        }

        /// Create a new full/empty storage in full state, initializing it with 
        /// the value provided
        full_empty(threadmanager::px_thread_self& self, value_type const& t)
        {
            set(self, t);
        }

        /// Destruct the full/empty data item
        ~full_empty()
        {
        }

        /// Atomically set the state to empty
        void empty(threadmanager::px_thread_self& self)
        {
        }
        
        /// Atomically set the state to full
        void fill(threadmanager::px_thread_self& self)
        {
        }
        
        /// Query the current state of the memory
        bool is_empty(threadmanager::px_thread_self& self)
        {
            return false;   // assume memory to be full (for now)
        }
        
        /// Wait for the memory to become full and then reads it, leaves memory
        /// in full state.
        ///
        /// \note When memory becomes full, all \a px_threads waiting for it
        ///       to become full with a read will receive the value at once and 
        ///       will be queued to run.
        value_type const& read(threadmanager::px_thread_self& self) const
        {
            self.yield(threadmanager::suspended);
            return get_data();
        }

        /// Wait for memory to become full and then set it to empty, returning 
        /// new value.
        ///
        /// \note When memory becomes full, only one thread blocked like this 
        ///       will be queued to run.
        value_type const& read_and_empty(threadmanager::px_thread_self& self) const
        {
            self.yield(threadmanager::suspended);
            return get_data();
        }

        /// Writes memory and atomically sets its state to full without waiting 
        /// for it to become empty.
        void set(threadmanager::px_thread_self& self, value_type const& data)
        {
            get_data() = data;
            fill(self);
        }

        /// Wait for memory to become empty, and then fill it.
        ///
        /// \note When memory becomes empty only one thread blocked like this 
        ///       will be queued to run.
        void write(threadmanager::px_thread_self& self, value_type const& data)
        {
            get_data() = data;
            fill(self);
        }

    private:
        // type safe accessors to the stored data
        typedef typename boost::add_pointer<value_type>::type pointer;
        typedef typename boost::add_pointer<value_type const>::type const_pointer;
        
        typedef typename boost::call_traits<T>::reference reference;
        typedef typename boost::call_traits<T>::const_reference const_reference;

        reference get_data()
        {
            return *static_cast<pointer>(data_.address());
        }
        const_reference get_data() const
        {
            return *static_cast<const_pointer>(data_.address());
        }

        // the stored data needs to be properly aligned
        typedef boost::aligned_storage<sizeof(value_type),
            boost::alignment_of<value_type>::value> storage_type;

        storage_type data_;
    };
    
}}

#endif


