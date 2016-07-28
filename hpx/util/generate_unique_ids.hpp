//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_GENERATE_UNIQUE_IDS_MAR_24_2008_1014AM)
#define HPX_UTIL_GENERATE_UNIQUE_IDS_MAR_24_2008_1014AM

#include <hpx/config.hpp>
#include <hpx/runtime/naming/address.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#include <memory>

#if defined(HPX_MSVC)
#pragma warning(push)
#pragma warning(disable: 4251)
#endif

namespace hpx { namespace util
{
    namespace detail
    {
        struct unique_id_ranges_base
        {
            unique_id_ranges_base(bool local_only_address_mode)
              : local_only_address_mode_(local_only_address_mode)
            {}

            virtual ~unique_id_ranges_base() {}

            virtual naming::gid_type get_id(std::size_t count,
                naming::address const& addr) = 0;
            virtual void set_range(naming::gid_type const& lower,
                naming::gid_type const& upper) = 0;

            bool local_only_address_mode() const
            {
                return local_only_address_mode_;
            }

            bool local_only_address_mode_;
        };
    }

    /// The unique_id_ranges class is a type responsible for generating
    /// unique ids for components, parcels, threads etc.
    class HPX_EXPORT unique_id_ranges
    {
    public:
        unique_id_ranges()
          : data_(create_implementation())
        {}

        /// Generate next unique component id
        naming::gid_type get_id(std::size_t count, naming::address const& addr)
        {
            return data_->get_id(count, addr);
        }

        void set_range(naming::gid_type const& lower,
            naming::gid_type const& upper)
        {
            data_->set_range(lower, upper);
        }

        bool local_only_address_mode() const
        {
            return data_->local_only_address_mode();
        }

    private:
        std::shared_ptr<detail::unique_id_ranges_base> create_implementation();

    private:
        std::shared_ptr<detail::unique_id_ranges_base> data_;
    };
}}

#if defined(HPX_MSVC)
#pragma warning(pop)
#endif

#endif


