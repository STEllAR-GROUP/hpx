//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/generate_unique_ids.hpp>
#include <hpx/util/unlock_guard.hpp>

#include <memory>
#include <mutex>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    // 'normal' distributed operation
    struct unique_id_ranges : unique_id_ranges_base
    {
        typedef hpx::lcos::local::spinlock mutex_type;

        mutex_type mtx_;

        /// size of the id range returned by command_getidrange
        /// FIXME: is this a policy?
        enum { range_delta = 0x100000 };

    public:
        unique_id_ranges()
          : mtx_(), lower_(0), upper_(0)
        {}

        /// Generate next unique component id
        naming::gid_type get_id(std::size_t count,
            naming::address::address_type addr)
        {
            // create a new id
            std::unique_lock<mutex_type> l(mtx_);

            // ensure next_id doesn't overflow
            while (!lower_ || (lower_ + count) > upper_)
            {
                lower_ = naming::invalid_gid;

                naming::gid_type lower;
                std::size_t count_ = (std::max)(std::size_t(range_delta), count);

                {
                    unlock_guard<std::unique_lock<mutex_type> > ul(l);
                    lower = hpx::agas::get_next_id(count_, 0);
                }

                // we ignore the result if some other thread has already set the
                // new lower range
                if (!lower_)
                {
                    lower_ = lower;
                    upper_ = lower + count_;
                }
            }

            naming::gid_type result = lower_;
            lower_ += count;
            return result;
        }

        void set_range(naming::gid_type const& lower,
            naming::gid_type const& upper)
        {
            std::lock_guard<mutex_type> l(mtx_);
            lower_ = lower;
            upper_ = upper;
        }

    private:
        /// The range of available ids for components
        naming::gid_type lower_;
        naming::gid_type upper_;
    };

    ///////////////////////////////////////////////////////////////////////
    // 'local' operation
    struct unique_id_ranges_local : unique_id_ranges_base
    {
    public:
        unique_id_ranges_local() {}

        /// Generate next unique component id
        naming::gid_type get_id(std::size_t count,
            naming::address::address_type addr)
        {
            return hpx::agas::get_next_id(count, addr);
        }

        void set_range(naming::gid_type const& lower,
            naming::gid_type const& upper)
        {}
    };
}}}

namespace hpx { namespace util
{
    std::shared_ptr<detail::unique_id_ranges_base>
    unique_id_ranges::create_implementation()
    {
        if (hpx::get_runtime().get_config().run_purely_local_agas())
        {
            return std::static_pointer_cast<detail::unique_id_ranges_base>(
                std::make_shared<detail::unique_id_ranges_local>());
        }

        return std::static_pointer_cast<detail::unique_id_ranges_base>(
            std::make_shared<detail::unique_id_ranges>());
    }
}}

