////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2008-2009 Chirag Dekate, Anshul Tandon
//  Copyright (c) 2012-2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_E43E0AF0_8A9D_4870_8CC7_E5AD53EF4798)
#define HPX_E43E0AF0_8A9D_4870_8CC7_E5AD53EF4798

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/address.hpp>

#include <boost/thread.hpp>
#include <boost/dynamic_bitset.hpp>

#include <string>
#include <vector>
#include <iosfwd>
#include <limits>

namespace hpx { namespace threads
{
    /// \cond NOINTERNAL
#if !defined(HPX_HAVE_MORE_THAN_64_THREADS)
    typedef boost::uint64_t mask_type;
    typedef boost::uint64_t mask_cref_type;

    static boost::uint64_t const bits[] =
    {
        0x0000000000000001ull, 0x0000000000000002ull, 0x0000000000000004ull, 0x0000000000000008ull,
        0x0000000000000010ull, 0x0000000000000020ull, 0x0000000000000040ull, 0x0000000000000080ull,
        0x0000000000000100ull, 0x0000000000000200ull, 0x0000000000000400ull, 0x0000000000000800ull,
        0x0000000000001000ull, 0x0000000000002000ull, 0x0000000000004000ull, 0x0000000000008000ull,
        0x0000000000010000ull, 0x0000000000020000ull, 0x0000000000040000ull, 0x0000000000080000ull,
        0x0000000000100000ull, 0x0000000000200000ull, 0x0000000000400000ull, 0x0000000000800000ull,
        0x0000000001000000ull, 0x0000000002000000ull, 0x0000000004000000ull, 0x0000000008000000ull,
        0x0000000010000000ull, 0x0000000020000000ull, 0x0000000040000000ull, 0x0000000080000000ull,
        0x0000000100000000ull, 0x0000000200000000ull, 0x0000000400000000ull, 0x0000000800000000ull,
        0x0000001000000000ull, 0x0000002000000000ull, 0x0000004000000000ull, 0x0000008000000000ull,
        0x0000010000000000ull, 0x0000020000000000ull, 0x0000040000000000ull, 0x0000080000000000ull,
        0x0000100000000000ull, 0x0000200000000000ull, 0x0000400000000000ull, 0x0000800000000000ull,
        0x0001000000000000ull, 0x0002000000000000ull, 0x0004000000000000ull, 0x0008000000000000ull,
        0x0010000000000000ull, 0x0020000000000000ull, 0x0040000000000000ull, 0x0080000000000000ull,
        0x0100000000000000ull, 0x0200000000000000ull, 0x0400000000000000ull, 0x0800000000000000ull,
        0x1000000000000000ull, 0x2000000000000000ull, 0x4000000000000000ull, 0x8000000000000000ull
    };

    inline bool any(mask_cref_type mask)
    {
        return (mask != 0) ? true : false;
    }

    inline mask_type not(mask_cref_type mask)
    {
        return ~mask;
    }

    inline bool test(mask_cref_type mask, std::size_t idx)
    {
        BOOST_ASSERT(idx < sizeof(bits)/sizeof(bits[0]));
        return (bits[idx] & mask) ? true : false;
    }

    inline void set(mask_type& mask, std::size_t idx)
    {
        BOOST_ASSERT(idx < sizeof(bits)/sizeof(bits[0]));
        mask |= bits[idx];
    }

    inline std::size_t mask_size(mask_cref_type mask)
    {
        return sizeof(bits)/sizeof(bits[0]);
    }

    inline void resize(mask_type& mask, std::size_t s)
    {
        BOOST_ASSERT(s <= sizeof(bits)/sizeof(bits[0]));
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
        return std::size_t(-1);
    }
#else
    typedef boost::dynamic_bitset<boost::uint64_t> mask_type;
    typedef boost::dynamic_bitset<boost::uint64_t> const& mask_cref_type;

    inline bool any(mask_cref_type mask)
    {
        return mask.any();
    }

    inline mask_type not(mask_cref_type mask)
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
        return mask.resize(s);
    }

    inline std::size_t find_first(mask_cref_type mask)
    {
        return mask.find_first();
    }
#endif
    /// \endcond

    struct topology
    {
        virtual ~topology() {}

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
            bool numa_sensitive, error_code& ec = throws) const = 0;

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
        virtual mask_cref_type get_thread_affinity_mask_from_lva(
            naming::address::address_type, error_code& ec = throws) const = 0;

        /// \brief Reduce thread priority of the current thread.
        ///
        /// \param ec         [in,out] this represents the error status on exit,
        ///                   if this is pre-initialized to \a hpx#throws
        ///                   the function will throw on error instead.
        virtual bool reduce_thread_priority(error_code& ec = throws) const;
    };

    HPX_API_EXPORT std::size_t hardware_concurrency();

    HPX_API_EXPORT topology const& get_topology();

#if defined(HPX_HAVE_HWLOC)
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        typedef std::vector<boost::int64_t> bounds_type;

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
                        BOOST_ASSERT(min >= 0);
                        index_bounds_.push_back(min);
                        index_bounds_.push_back(max);
                    }
                }
            }

            bool operator==(spec_type const& rhs) const
            {
                if (type_ != rhs.type_ || index_bounds_.size() != rhs.index_bounds_.size())
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
        typedef std::vector<full_mapping_type> mappings_type;

        HPX_API_EXPORT bounds_type extract_bounds(spec_type& m,
            std::size_t default_last, error_code& ec);

        HPX_API_EXPORT void parse_mappings(std::string const& spec,
            mappings_type& mappings, error_code& ec = throws);
    }

    HPX_API_EXPORT void parse_affinity_options(std::string const& spec,
        std::vector<mask_type>& affinities, error_code& ec = throws);

    HPX_API_EXPORT void print_affinity_options(std::ostream& s,
        std::size_t num_threads, std::string const& affinity_options,
        error_code& ec = throws);
#endif

    /// \endcond
}}

#endif // HPX_E43E0AF0_8A9D_4870_8CC7_E5AD53EF4798

