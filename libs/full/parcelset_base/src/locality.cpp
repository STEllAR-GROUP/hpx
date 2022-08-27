//  Copyright (c)      2014 Thomas Heller
//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/serialization.hpp>
#include <hpx/modules/type_support.hpp>
#include <hpx/modules/util.hpp>

#include <hpx/parcelset_base/locality.hpp>
#include <hpx/parcelset_base/locality_interface.hpp>

#include <string>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parcelset {

    locality::locality(locality const& other)
      : impl_(other.impl_ ? other.impl_->clone() : nullptr)
    {
    }

    locality::locality(locality&& other) noexcept
      : impl_(other.impl_ ? other.impl_->move() : nullptr)
    {
    }

    locality& locality::operator=(locality const& other)
    {
        if (this != &other)
        {
            if (other.impl_)
            {
                impl_.reset(other.impl_->clone());
            }
            else
            {
                impl_.reset();
            }
        }
        return *this;
    }

    locality& locality::operator=(locality&& other) noexcept
    {
        if (this != &other)
        {
            if (other.impl_)
            {
                impl_.reset(other.impl_->move());
            }
            else
            {
                impl_.reset();
            }
        }
        return *this;
    }

    ///////////////////////////////////////////////////////////////////////////
    bool operator==(locality const& lhs, locality const& rhs)
    {
        if (lhs.impl_ == rhs.impl_)
            return true;
        if (!lhs.impl_ || !rhs.impl_)
            return false;
        return lhs.impl_->equal(*rhs.impl_);
    }

    bool operator!=(locality const& lhs, locality const& rhs)
    {
        return !(lhs == rhs);
    }

    bool operator<(locality const& lhs, locality const& rhs)
    {
        if (lhs.impl_ == rhs.impl_)
            return false;
        if (!lhs.impl_ || !rhs.impl_)
            return false;
        return lhs.impl_->less_than(*rhs.impl_);
    }

    bool operator>(locality const& lhs, locality const& rhs)
    {
        if (lhs.impl_ == rhs.impl_)
            return false;
        if (!lhs.impl_ || !rhs.impl_)
            return false;
        return !(lhs < rhs) && !(lhs == rhs);
    }

    ///////////////////////////////////////////////////////////////////////////
    std::ostream& operator<<(std::ostream& os, locality const& l)
    {
        if (!l.impl_)
            return os;
        return l.impl_->print(os);
    }

    ///////////////////////////////////////////////////////////////////////////
    void locality::save(
        serialization::output_archive& ar, const unsigned int) const
    {
#if defined(HPX_HAVE_NETWORKING)
        std::string t = type();
        ar << t;
        if (t.empty())
        {
            return;
        }

        impl_->save(ar);
#else
        HPX_THROW_EXCEPTION(invalid_status, "locality::save",
            "this shouldn't be called if networking is disabled");
        HPX_UNUSED(ar);
#endif
    }

    void locality::load(serialization::input_archive& ar, const unsigned int)
    {
#if defined(HPX_HAVE_NETWORKING)
        std::string t;
        ar >> t;
        if (t.empty())
        {
            return;
        }

        impl_ = HPX_MOVE(create_locality(t).impl_);
        impl_->load(ar);
        HPX_ASSERT(impl_->valid());
#else
        HPX_THROW_EXCEPTION(invalid_status, "locality::load",
            "this shouldn't be called if networking is disabled");
        HPX_UNUSED(ar);
#endif
    }

    std::ostream& operator<<(std::ostream& os, endpoints_type const& endpoints)
    {
        hpx::util::ios_flags_saver ifs(os);
        os << "[ ";
        for (endpoints_type::value_type const& loc : endpoints)
        {
            os << "(" << loc.first << ":" << loc.second << ") ";
        }
        os << "]";

        return os;
    }
}}    // namespace hpx::parcelset
