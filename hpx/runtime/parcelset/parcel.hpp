//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2015-2016 Thomas Heller
//  Copyright (c) 2007 Richard D Guidry Jr
//  Copyright (c) 2007 Alexandre (aka Alex) TABBAL
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_PARCEL_MAR_26_2008_1051AM)
#define HPX_PARCELSET_PARCEL_MAR_26_2008_1051AM

#include <hpx/config.hpp>
#include <hpx/runtime/actions_fwd.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/parcelset_fwd.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace serialization
{
    struct binary_filter;
}}
namespace hpx { namespace parcelset { namespace policies
{
    struct message_handler;
}}}

namespace hpx { namespace parcelset
{
    namespace detail
    {
        struct parcel_data
        {
        public:
            parcel_data();
            parcel_data(naming::gid_type&& dest, naming::address&& addr,
                bool has_continuation);
            parcel_data(parcel_data && rhs);
            parcel_data& operator=(parcel_data && rhs);
            template <typename Archive>
            void serialize(Archive &ar, unsigned);

#if defined(HPX_HAVE_PARCEL_PROFILING)
            naming::gid_type parcel_id_;
            double start_time_;
            double creation_time_;
#endif

            naming::gid_type source_id_;
            naming::gid_type dest_;
            naming::address addr_;

            bool has_continuation_;
        };
    }

    class HPX_EXPORT parcel
    {

    private:

        typedef
            std::map<const naming::gid_type*, naming::gid_type>
            split_gids_type;

#if defined(HPX_DEBUG)
        bool is_valid() const;
#else
        // Only used in debug mode.
        bool is_valid() const
        {
            return true;
        }
#endif

    public:
        parcel();
        ~parcel();

    private:
        parcel(
            naming::gid_type&& dest,
            naming::address&& addr,
            std::unique_ptr<actions::base_action> act
        );

        friend struct detail::create_parcel;

    public:
        parcel(parcel && other);
        parcel &operator=(parcel && other);

        void reset();

        actions::base_action *get_action() const;

        naming::id_type source_id() const;

        void set_source_id(naming::id_type const & source_id);

        void set_destination_id(naming::gid_type&& dest);

        naming::gid_type const& destination() const;

        naming::address const& addr() const;

        naming::address& addr();

        std::uint32_t destination_locality_id() const;

        naming::gid_type const& destination_locality() const;

        double start_time() const;

        void set_start_time(double time);

        double creation_time() const;

        threads::thread_priority get_thread_priority() const;

#if defined(HPX_HAVE_PARCEL_PROFILING)
        naming::gid_type const parcel_id() const;

        naming::gid_type & parcel_id();
#endif

        serialization::binary_filter* get_serialization_filter() const;

        policies::message_handler* get_message_handler(
            parcelset::parcelhandler* ph, locality const& loc) const;

        bool does_termination_detection() const;

        split_gids_type& split_gids() const;

        void set_split_gids(split_gids_type&& split_gids);

        std::size_t const& num_chunks() const;

        std::size_t & num_chunks();

        std::size_t const& size() const;

        std::size_t & size();

        void schedule_action(std::size_t num_thread = std::size_t(-1));

        // returns true if parcel was migrated, false if scheduled locally
        bool load_schedule(serialization::input_archive & ar,
            std::size_t num_thread, bool& deferred_schedule);

        // generate unique parcel id
        static naming::gid_type generate_unique_id(
            std::uint32_t locality_id = naming::invalid_locality_id);

    private:
        friend std::ostream& operator<< (std::ostream& os, parcel const& req);

        // serialization support
        friend class hpx::serialization::access;
        void load_data(serialization::input_archive & ar);
        void serialize(serialization::input_archive & ar, unsigned);
        void serialize(serialization::output_archive & ar, unsigned);

        naming::address_type determine_lva();

        detail::parcel_data data_;
        std::unique_ptr<actions::base_action> action_;

        split_gids_type split_gids_;
        std::size_t size_;
        std::size_t num_chunks_;
    };

    HPX_EXPORT std::string dump_parcel(parcel const& p);
}}

HPX_IS_BITWISE_SERIALIZABLE(hpx::parcelset::detail::parcel_data);

#include <hpx/config/warnings_suffix.hpp>

#endif
