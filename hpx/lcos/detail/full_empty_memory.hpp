//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_FULLEMPTYMEMORY_JUN_16_2008_1102AM)
#define HPX_UTIL_FULLEMPTYMEMORY_JUN_16_2008_1102AM

#include <boost/noncopyable.hpp>
#include <boost/call_traits.hpp>
#include <hpx/util/move.hpp>

#include <hpx/lcos/detail/full_empty_entry.hpp>

namespace hpx { namespace lcos { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
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
    /// If a full_empty memory location goes out of scope while reads/writes are
    /// waiting on it, those operations will be resumed immediately, and they
    /// either throw a \a hpx#yield_aborted exception or set their
    /// \a hpx#error_code parameter to \a hpx#yield_aborted.
    ///
    /// \tparam T   The template parameter \a T defines the type of the memory
    ///             location to be guarded by an empty/full bit. It is possible
    ///             to use any C++ data type with the empty/full mechanism.
    ///             If you want to use the empty/full synchronization facilities
    ///             without having to transfer (read/write) any data you can
    ///             use the specialization lcos#full_empty<void>.
    template <typename T>
    class full_empty
    {
    private:
        typedef T value_type;

    public:
//         typedef typename detail::full_empty_entry<T>::mutex_type mutex_type;

        /// \brief Create a new full/empty storage in empty state
        full_empty()
        {}

        template <typename T0>
        full_empty(BOOST_FWD_REF(T0) t0)
          : data_(boost::forward<T0>(t0))
        {}

        /// \brief Destruct the full/empty data item
        ~full_empty()
        {}

        /// \brief Atomically set the state to empty without releasing any
        ///        waiting \a threads. This function is mainly usable for
        ///        initialization and debugging purposes.
        ///
        /// \note    This function will create a new full/empty entry in the
        ///          store if it doesn't exist yet.
        void set_empty(error_code& ec = throws)
        {
            data_.set_empty(ec);
        }

        /// \brief Atomically set the state to full without releasing any
        ///        waiting \a threads. This function is mainly usable for
        ///        initialization and debugging purposes.
        ///
        /// \note    This function will not create a new full/empty entry in
        ///          the store if it doesn't exist yet.
        void set_full(error_code& ec = throws)
        {
            data_.set_full(ec);
        }

        /// \brief Query the current state of the memory
        bool is_empty() const
        {
            return data_.is_empty();
        }

        /// \brief  Waits for the memory to become full and then reads it,
        ///         leaves memory in full state. If the location is empty the
        ///         calling thread will wait (block) for another thread to call
        ///         either the function \a set or the function \a write.
        ///
        /// \param ec [in,out] this represents the error status on exit,
        ///           if this is pre-initialized to \a hpx#throws
        ///           the function will throw on error instead. If the operation
        ///           blocks and is aborted because the object went out of
        ///           scope, the code \a hpx#yield_aborted is set or thrown.
        ///
        /// \note   When memory becomes full, all \a threads waiting for it
        ///         to become full with a read will receive the value at once
        ///         and will be queued to run.
        template <typename Target>
        void read(Target& dest, error_code& ec = throws)
        {
            data_.enqueue_full_full(dest, ec);
        }

        /// \brief  Waits for memory to become full and then reads it, sets
        ///         memory to empty. If the location is empty the calling
        ///         thread will wait (block) for another thread to call either
        ///         the function \a set or the function \a write.
        ///
        /// \param ec [in,out] this represents the error status on exit,
        ///           if this is pre-initialized to \a hpx#throws
        ///           the function will throw on error instead. If the operation
        ///           blocks and is aborted because the object went out of
        ///           scope, the code \a hpx#yield_aborted is set or thrown.
        ///
        /// \note   When memory becomes empty, only one thread blocked like this
        ///         will be queued to run (one thread waiting in a \a write
        ///         function).
        template <typename Target>
        void read_and_empty(Target& dest, error_code& ec = throws)
        {
            data_.enqueue_full_empty(dest, ec);
        }

        /// \brief  Waits for memory to become full and then reads it, sets
        ///         memory to empty if the predicate is true. If the location
        ///         is empty the calling thread will wait (block) for another
        ///         thread to call either the function \a set or the function
        ///         \a write.
        ///
        /// \param pred [in] this predicate will be invoked for the rerieved
        ///           data value. I the predicate returns true, the memory
        ///           will be set to empty.
        /// \param ec [in,out] this represents the error status on exit,
        ///           if this is pre-initialized to \a hpx#throws
        ///           the function will throw on error instead. If the operation
        ///           blocks and is aborted because the object went out of
        ///           scope, the code \a hpx#yield_aborted is set or thrown.
        ///
        /// \note   When memory becomes empty, only one thread blocked like this
        ///         will be queued to run (one thread waiting in a \a write
        ///         function).
        template <typename Target, typename F>
        void read_and_empty_if(Target& dest, F const& f, error_code& ec = throws)
        {
            data_.enqueue_full_empty_if(dest, f, ec);
        }

        /// \brief  Writes memory and atomically sets its state to full without
        ///         waiting for it to become empty.
        ///
        /// \note   Even if the function itself doesn't block, setting the
        ///         location to full using \a set might re-activate threads
        ///         waiting on this in a \a read or \a read_and_empty function.
        template <typename Target>
        void set(BOOST_FWD_REF(Target) data)
        {
            data_.set_and_fill(boost::forward<Target>(data));
        }

        /// \brief  Waits for memory to become empty, and then fills it. If the
        ///         location is filled the calling thread will wait (block) for
        ///         another thread to call the function \a read_and_empty.
        ///
        /// \param ec [in,out] this represents the error status on exit,
        ///           if this is pre-initialized to \a hpx#throws
        ///           the function will throw on error instead. If the operation
        ///           blocks and is aborted because the object went out of
        ///           scope, the code \a hpx#yield_aborted is set or thrown.
        ///
        /// \note   When memory becomes empty only one thread blocked like this
        ///         will be queued to run.
        template <typename Target>
        void write(BOOST_FWD_REF(Target) data, error_code& ec = throws)
        {
            data_.enqueue_if_full(boost::forward<Target>(data), ec);
        }

        /// \brief  Calls the supplied function passing along the stored data
        ///         (if full)
        ///
        /// \param f [in] The function to be called
        ///
        /// \returns This function returns \a false if the FE memory is empty
        ///          otherwise it returns the return value of \p f.
        template <typename F>
        bool peek(F f) const
        {
            return data_.peek(f);
        }

//         /// Retrieve the mutex which protects the underlying data store
//         mutex_type& get_mutex() const
//         {
//             return data_.get_mutex();
//         }

    private:
        full_empty_entry<T> data_;
    };

}}}

#endif


