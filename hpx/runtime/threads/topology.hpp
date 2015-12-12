////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2012-2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_E43E0AF0_8A9D_4870_8CC7_E5AD53EF4798)
#define HPX_E43E0AF0_8A9D_4870_8CC7_E5AD53EF4798

#include <hpx/config.hpp>
#include <hpx/exception_fwd.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/util/assert.hpp>

#include <boost/thread.hpp>
#include <boost/variant.hpp>
#include <boost/dynamic_bitset.hpp>

#include <string>
#include <vector>
#include <iosfwd>
#include <limits>
#if defined(HPX_HAVE_MORE_THAN_64_THREADS) || (defined(HPX_HAVE_MAX_CPU_COUNT) \
            && HPX_HAVE_MAX_CPU_COUNT > 64)
#include <bitset>
#endif

namespace hpx { namespace threads
{
    /// \cond NOINTERNAL
#if !defined(HPX_HAVE_MORE_THAN_64_THREADS) || (defined(HPX_HAVE_MAX_CPU_COUNT) \
             && HPX_HAVE_MAX_CPU_COUNT <= 64)
    typedef boost::uint64_t mask_type;
    typedef boost::uint64_t mask_cref_type;

    inline boost::uint64_t bits(std::size_t idx)
    {
       HPX_ASSERT(idx < CHAR_BIT * sizeof(mask_type));
       return boost::uint64_t(1) << idx;
    }

    inline bool any(mask_cref_type mask)
    {
        return mask != 0;
    }

    inline mask_type not_(mask_cref_type mask)
    {
        return ~mask;
    }

    inline bool test(mask_cref_type mask, std::size_t idx)
    {
        HPX_ASSERT(idx < CHAR_BIT * sizeof(mask_type));
        return (bits(idx) & mask) != 0;
    }

    inline void set(mask_type& mask, std::size_t idx)
    {
        HPX_ASSERT(idx < CHAR_BIT * sizeof(mask_type));
        mask |= bits(idx);
    }

    inline std::size_t mask_size(mask_cref_type mask)
    {
        return CHAR_BIT * sizeof(mask_type);
    }

    inline void resize(mask_type& mask, std::size_t s)
    {
        HPX_ASSERT(s <= CHAR_BIT * sizeof(mask_type));
    }

    inline std::size_t find_first(mask_cref_type mask)
    {
        if (mask) {
            std::size_t c = 0;    // Will count mask's trailing zero bits.

            // Set mask's trailing 0s to 1s and zero rest.
            mask = (mask ^ (mask - 1)) >> 1;
            for (/**/; mask; ++c)
                mask >>= 1;

            return c;
        }
        return ~std::size_t(0);
    }

    inline bool equal(mask_cref_type lhs, mask_cref_type rhs, std::size_t)
    {
        return lhs == rhs;
    }

    // return true if at least one of the masks has a bit set
    inline bool bit_or(mask_cref_type lhs, mask_cref_type rhs, std::size_t)
    {
        return (lhs | rhs) != 0;
    }

    // return true if at least one bit is set in both masks
    inline bool bit_and(mask_cref_type lhs, mask_cref_type rhs, std::size_t)
    {
        return (lhs & rhs) != 0;
    }

#define HPX_CPU_MASK_PREFIX "0x"

#else
# if defined(HPX_HAVE_MAX_CPU_COUNT)
    typedef std::bitset<HPX_HAVE_MAX_CPU_COUNT> mask_type;
    typedef std::bitset<HPX_HAVE_MAX_CPU_COUNT> const& mask_cref_type;
# else
    typedef boost::dynamic_bitset<boost::uint64_t> mask_type;
    typedef boost::dynamic_bitset<boost::uint64_t> const& mask_cref_type;
# endif

    inline bool any(mask_cref_type mask)
    {
        return mask.any();
    }

    inline mask_type not_(mask_cref_type mask)
    {
        return ~mask;
    }

    inline bool test(mask_cref_type mask, std::size_t idx)
    {
        return mask.test(idx);
    }

    inline void set(mask_type& mask, std::size_t idx)
    {
        mask.set(idx);
    }

    inline std::size_t mask_size(mask_cref_type mask)
    {
        return mask.size();
    }

    inline void resize(mask_type& mask, std::size_t s)
    {
# if defined(HPX_HAVE_MAX_CPU_COUNT)
        HPX_ASSERT(s <= mask.size());
# else
        return mask.resize(s);
# endif
    }

    inline std::size_t find_first(mask_cref_type mask)
    {
# if defined(HPX_HAVE_MAX_CPU_COUNT)
        if (mask.any())
        {
            for (std::size_t i = 0; i != HPX_HAVE_MAX_CPU_COUNT; ++i)
            {
                if (mask[i])
                    return i;
            }
        }
        return ~std::size_t(0);
# else
        return mask.find_first();
# endif
    }

# if defined(HPX_HAVE_MAX_CPU_COUNT)
#define HPX_CPU_MASK_PREFIX "0b"
#else
#define HPX_CPU_MASK_PREFIX "0x"
#endif

    inline bool equal(mask_cref_type lhs, mask_cref_type rhs, std::size_t numbits)
    {
        for (std::size_t j = 0; j != numbits; ++j)
        {
            if (test(lhs, j) != test(rhs, j))
            {
                return false;
            }
        }
        return true;
    }

    // return true if at least one of the masks has a bit set
    inline bool bit_or(mask_cref_type lhs, mask_cref_type rhs, std::size_t numbits)
    {
        for (std::size_t j = 0; j != numbits; ++j)
        {
            if (test(lhs, j) || test(rhs, j))
            {
                return true;
            }
        }
        return false;
    }

    // return true if at least one bit is set in both masks
    inline bool bit_and(mask_cref_type lhs, mask_cref_type rhs, std::size_t numbits)
    {
        for (std::size_t j = 0; j != numbits; ++j)
        {
            if (test(lhs, j) && test(rhs, j))
            {
                return true;
            }
        }
        return false;
    }
#endif
    /// \endcond

    struct topology
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
        virtual void* allocate(std::size_t len) = 0;

        /// Free memory that was previously allocated by allocate
        virtual void deallocate(void* addr, std::size_t len) = 0;
    };

    HPX_API_EXPORT std::size_t hardware_concurrency();

    HPX_API_EXPORT topology const& get_topology();

#if defined(HPX_HAVE_HWLOC)
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        typedef std::vector<boost::int64_t> bounds_type;

        enum distribution_type
        {
            compact  = 0x01,
            scatter  = 0x02,
            balanced = 0x04
        };

        struct spec_type
        {
            enum type { unknown, thread, socket, numanode, core, pu };
            HPX_API_EXPORT static char const* type_name(type t);

            static boost::int64_t all_entities()
            {
                return (std::numeric_limits<boost::int64_t>::min)();
            }

            spec_type(type t = unknown, boost::int64_t min = all_entities(),
                    boost::int64_t max = all_entities())
              : type_(t), index_bounds_()
            {
                if (t != unknown) {
                    if (max == 0 || max == all_entities()) {
                        // one or all entities
                        index_bounds_.push_back(min);
                    }
                    else if (min != all_entities()) {
                        // all entities between min and -max, or just min,max
                        HPX_ASSERT(min >= 0);
                        index_bounds_.push_back(min);
                        index_bounds_.push_back(max);
                    }
                }
            }

            bool operator==(spec_type const& rhs) const
            {
                if (type_ != rhs.type_ || index_bounds_.size() !=
                    rhs.index_bounds_.size())
                    return false;

                for (std::size_t i = 0; i < index_bounds_.size(); ++i)
                {
                    if (index_bounds_[i] != rhs.index_bounds_[i])
                        return false;
                }

                return true;
            }

            type type_;
            bounds_type index_bounds_;
        };

        typedef std::vector<spec_type> mapping_type;
        typedef std::pair<spec_type, mapping_type> full_mapping_type;
        typedef std::vector<full_mapping_type> mappings_spec_type;
        typedef boost::variant<distribution_type, mappings_spec_type> mappings_type;

        HPX_API_EXPORT bounds_type extract_bounds(spec_type& m,
            std::size_t default_last, error_code& ec);

        HPX_API_EXPORT void parse_mappings(std::string const& spec,
            mappings_type& mappings, error_code& ec = throws);
    }

    HPX_API_EXPORT void parse_affinity_options(std::string const& spec,
        std::vector<mask_type>& affinities,
        std::size_t used_cores,
        std::size_t max_cores,
        std::vector<std::size_t>& num_pus,
        error_code& ec = throws);

    // backwards compatibility helper
    inline void parse_affinity_options(std::string const& spec,
        std::vector<mask_type>& affinities,
        error_code& ec = throws)
    {
        std::vector<std::size_t> num_pus;
        parse_affinity_options(spec, affinities, 1, 1, num_pus, ec);
    }
#endif

    /// \endcond
}}

#endif // HPX_E43E0AF0_8A9D_4870_8CC7_E5AD53EF4798

