////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2012-2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#ifndef HPX_RUNTIME_THREADS_TOPOLOGY_HPP
#define HPX_RUNTIME_THREADS_TOPOLOGY_HPP

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/threads/cpu_mask.hpp>

#include <boost/thread/thread.hpp>

#include <cstddef>
#include <iosfwd>
#include <utility>

#if defined(HPX_HAVE_HWLOC)
#include <hpx/error_code.hpp>

#include <cstddef>
#include <string>
#include <vector>
#endif

namespace hpx { namespace threads
{
    struct HPX_EXPORT topology
    {
        virtual ~topology() {}

        virtual std::size_t get_pu_number(
            std::size_t num_thread
          , error_code& ec = throws
            ) const = 0;

        /// \brief Return the NUMA node number of the processing unit the
        ///        given thread is running on.
        ///
        /// \param ec         [in,out] this represents the error status on exit,
        ///                   if this is pre-initialized to \a hpx#throws
        ///                   the function will throw on error instead.
        virtual std::size_t get_numa_node_number(std::size_t num_thread,
            error_code& ec = throws) const = 0;

        /// \brief Return a bit mask where each set bit corresponds to a
        ///        processing unit available to the application.
        ///
        /// \param ec         [in,out] this represents the error status on exit,
        ///                   if this is pre-initialized to \a hpx#throws
        ///                   the function will throw on error instead.
        virtual mask_cref_type get_machine_affinity_mask(
            error_code& ec = throws) const = 0;

        /// \brief Return a bit mask where each set bit corresponds to a
        ///        processing unit available to the service threads in the
        ///        application.
        ///
        /// \param used_processing_units [in] This is the mask of processing
        ///                   units which are not available for service threads.
        /// \param ec         [in,out] this represents the error status on exit,
        ///                   if this is pre-initialized to \a hpx#throws
        ///                   the function will throw on error instead.
        virtual mask_type get_service_affinity_mask(
            mask_cref_type used_processing_units, error_code& ec = throws) const;

        /// \brief Return a bit mask where each set bit corresponds to a
        ///        processing unit available to the given thread inside
        ///        the socket it is running on.
        ///
        /// \param ec         [in,out] this represents the error status on exit,
        ///                   if this is pre-initialized to \a hpx#throws
        ///                   the function will throw on error instead.
        virtual mask_cref_type get_socket_affinity_mask(std::size_t num_thread,
            bool numa_sensitive, error_code& ec = throws) const = 0;

        /// \brief Return a bit mask where each set bit corresponds to a
        ///        processing unit available to the given thread inside
        ///        the NUMA domain it is running on.
        ///
        /// \param ec         [in,out] this represents the error status on exit,
        ///                   if this is pre-initialized to \a hpx#throws
        ///                   the function will throw on error instead.
        virtual mask_cref_type get_numa_node_affinity_mask(std::size_t num_thread,
            bool numa_sensitive, error_code& ec = throws) const = 0;

        /// \brief Return a bit mask where each set bit corresponds to a
        ///        processing unit associated with the given NUMA node.
        ///
        /// \param ec         [in,out] this represents the error status on exit,
        ///                   if this is pre-initialized to \a hpx#throws
        ///                   the function will throw on error instead.
        virtual mask_type get_numa_node_affinity_mask_from_numa_node(
            std::size_t num_node) const = 0;

        /// \brief Return a bit mask where each set bit corresponds to a
        ///        processing unit available to the given thread inside
        ///        the core it is running on.
        ///
        /// \param ec         [in,out] this represents the error status on exit,
        ///                   if this is pre-initialized to \a hpx#throws
        ///                   the function will throw on error instead.
        virtual mask_cref_type get_core_affinity_mask(std::size_t num_thread,
            bool numa_sensitive, error_code& ec = throws) const = 0;

        /// \brief Return a bit mask where each set bit corresponds to a
        ///        processing unit available to the given thread.
        ///
        /// \param ec         [in,out] this represents the error status on exit,
        ///                   if this is pre-initialized to \a hpx#throws
        ///                   the function will throw on error instead.
        virtual mask_cref_type get_thread_affinity_mask(std::size_t num_thread,
            bool numa_sensitive = false, error_code& ec = throws) const = 0;

        /// \brief Use the given bit mask to set the affinity of the given
        ///        thread. Each set bit corresponds to a processing unit the
        ///        thread will be allowed to run on.
        ///
        /// \param ec         [in,out] this represents the error status on exit,
        ///                   if this is pre-initialized to \a hpx#throws
        ///                   the function will throw on error instead.
        ///
        /// \note  Use this function on systems where the affinity must be
        ///        set from outside the thread itself.
        virtual void set_thread_affinity_mask(boost::thread& t,
            mask_cref_type mask, error_code& ec = throws) const = 0;

        /// \brief Use the given bit mask to set the affinity of the given
        ///        thread. Each set bit corresponds to a processing unit the
        ///        thread will be allowed to run on.
        ///
        /// \param ec         [in,out] this represents the error status on exit,
        ///                   if this is pre-initialized to \a hpx#throws
        ///                   the function will throw on error instead.
        ///
        /// \note  Use this function on systems where the affinity must be
        ///        set from inside the thread itself.
        virtual void set_thread_affinity_mask(mask_cref_type mask,
            error_code& ec = throws) const = 0;

        /// \brief Return a bit mask where each set bit corresponds to a
        ///        processing unit co-located with the memory the given
        ///        address is currently allocated on.
        ///
        /// \param ec         [in,out] this represents the error status on exit,
        ///                   if this is pre-initialized to \a hpx#throws
        ///                   the function will throw on error instead.
        virtual mask_type get_thread_affinity_mask_from_lva(
            naming::address_type, error_code& ec = throws) const = 0;

        /// \brief Prints the \param m to os in a human readable form
        virtual void print_affinity_mask(std::ostream& os,
            std::size_t num_thread, mask_type const& m) const = 0;

        /// \brief Reduce thread priority of the current thread.
        ///
        /// \param ec         [in,out] this represents the error status on exit,
        ///                   if this is pre-initialized to \a hpx#throws
        ///                   the function will throw on error instead.
        virtual bool reduce_thread_priority(error_code& ec = throws) const;

        /// \brief Return the number of available NUMA domains
        virtual std::size_t get_number_of_sockets() const = 0;

        /// \brief Return the number of available NUMA domains
        virtual std::size_t get_number_of_numa_nodes() const = 0;

        /// \brief Return the number of available cores
        virtual std::size_t get_number_of_cores() const = 0;

        /// \brief Return the number of available hardware processing units
        virtual std::size_t get_number_of_pus() const = 0;

        /// \brief Return number of cores in given numa domain
        virtual std::size_t get_number_of_numa_node_cores(std::size_t numa) const = 0;

        /// \brief Return number of processing units in a given numa domain
        virtual std::size_t get_number_of_numa_node_pus(std::size_t numa) const = 0;

        /// \brief Return number of processing units in a given socket
        virtual std::size_t get_number_of_socket_pus(std::size_t socket) const = 0;

        /// \brief Return number of processing units in given core
        virtual std::size_t get_number_of_core_pus(std::size_t core) const = 0;

        virtual std::size_t get_core_number(std::size_t num_thread,
            error_code& ec = throws) const = 0;

        virtual mask_type get_cpubind_mask(error_code& ec = throws) const = 0;
        virtual mask_type get_cpubind_mask(boost::thread & handle,
            error_code& ec = throws) const = 0;

        virtual void write_to_log() const = 0;

        /// This is equivalent to malloc(), except that it tries to allocate
        /// page-aligned memory from the OS.
        virtual void* allocate(std::size_t len) const = 0;

        /// Free memory that was previously allocated by allocate
        virtual void deallocate(void* addr, std::size_t len) const = 0;
    };

    HPX_API_EXPORT std::size_t hardware_concurrency();

    HPX_API_EXPORT topology const& get_topology();

#if defined(HPX_HAVE_HWLOC)
    HPX_API_EXPORT void parse_affinity_options(std::string const& spec,
        std::vector<mask_type>& affinities,
        std::size_t used_cores,
        std::size_t max_cores,
        std::size_t num_threads,
        std::vector<std::size_t>& num_pus,
        error_code& ec = throws);

    // backwards compatibility helper
    inline void parse_affinity_options(std::string const& spec,
        std::vector<mask_type>& affinities, error_code& ec = throws)
    {
        std::vector<std::size_t> num_pus;
        parse_affinity_options(spec, affinities, 1, 1, affinities.size(),
            num_pus, ec);
    }
#endif

    /// \endcond
}}

#endif /*HPX_RUNTIME_THREADS_TOPOLOGY_HPP*/
