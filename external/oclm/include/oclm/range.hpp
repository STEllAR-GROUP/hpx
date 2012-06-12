
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_RANGE_HPP
#define OCLM_RANGE_HPP

#include <oclm/range.hpp>

namespace oclm {

    struct range
    {
        range() {}
        range(std::size_t v0) : values(1, v0) {}
        range(std::size_t v0, std::size_t v1) : values(2, v0)
        {
            values[1] = v1;
        }
        range(std::size_t v0, std::size_t v1, std::size_t v2) : values(3, v0)
        {
            values[1] = v1;
            values[2] = v2;
        }

        std::size_t dim() const
        {
            return values.size();
        }

        operator const std::size_t *()
        {
            return values.size() == 0 ? 0 : &values[0];
        }

        std::vector<std::size_t> values;
    };

    namespace tag
    {
        struct global_ {};
        struct local_ {};
        struct offset_ {};
        struct collection_ {};
    }

    template <typename Tag>
    struct kernel_range
    {
        range r;
    };

    inline kernel_range<tag::global_> global()
    {
        return kernel_range<tag::global_>();
    }

    inline kernel_range<tag::global_> global(std::size_t v0)
    {
        kernel_range<tag::global_> r = {range(v0)};
        return r;
    }

    inline kernel_range<tag::global_> global(std::size_t v0, std::size_t v1)
    {
        kernel_range<tag::global_> r = {range(v0, v1)};
        return r;
    }

    inline kernel_range<tag::global_> global(std::size_t v0, std::size_t v1, std::size_t v2)
    {
        kernel_range<tag::global_> r = {range(v0, v1, v2)};
        return r;
    }

    inline kernel_range<tag::local_> local()
    {
        return kernel_range<tag::local_>();
    }

    inline kernel_range<tag::local_> local(std::size_t v0)
    {
        kernel_range<tag::local_> r = {range(v0)};
        return r;
    }

    inline kernel_range<tag::local_> local(std::size_t v0, std::size_t v1)
    {
        kernel_range<tag::local_> r = {range(v0, v1)};
        return r;
    }

    inline kernel_range<tag::local_> local(std::size_t v0, std::size_t v1, std::size_t v2)
    {
        kernel_range<tag::local_> r = {range(v0, v1, v2)};
        return r;
    }

    inline kernel_range<tag::offset_> offset()
    {
        return kernel_range<tag::offset_>();
    }

    inline kernel_range<tag::offset_> offset(std::size_t v0)
    {
        kernel_range<tag::offset_> r = {range(v0)};
        return r;
    }

    inline kernel_range<tag::offset_> offset(std::size_t v0, std::size_t v1)
    {
        kernel_range<tag::offset_> r = {range(v0, v1)};
        return r;
    }

    inline kernel_range<tag::offset_> offset(std::size_t v0, std::size_t v1, std::size_t v2)
    {
        kernel_range<tag::offset_> r = {range(v0, v1, v2)};
        return r;
    }

    template <>
    struct kernel_range<tag::collection_>
    {
        kernel_range<tag::offset_> offset_r;
        kernel_range<tag::global_> global_r;
        kernel_range<tag::local_> local_r;

        void set(kernel_range<tag::offset_> const & r)
        {
            offset_r = r;
        }

        void set(kernel_range<tag::global_> const & r)
        {
            global_r = r;
        }

        void set(kernel_range<tag::local_> const & r)
        {
            local_r = r;
        }
        
        void set(kernel_range<tag::collection_> const & r)
        {
            offset_r = r.offset_r;
            global_r = r.global_r;
            local_r = r.local_r;
        }
    };

    typedef kernel_range<tag::collection_> ranges_type;

    template <typename Tag1, typename Tag2>
    kernel_range<tag::collection_>
    operator,(kernel_range<Tag1> const & r1, kernel_range<Tag2> const & r2)
    {
        kernel_range<tag::collection_> coll;
        coll.set(r1);
        coll.set(r2);
        return coll;
    }
}

#endif
