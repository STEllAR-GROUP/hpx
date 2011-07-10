

#include <hpx/hpx_fwd.hpp>
#include <hpx/example/geometry/point.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_free.hpp>

namespace hpx { namespace geometry
{
 
///////////////////////////////////////////////////////////////////////////
    // declarations only
    template <typename Archive>
    void save(Archive& ar, hpx::geometry::point const& ep, unsigned int);

    template <typename Archive>
    void load(Archive& ar, hpx::geometry::point& e, unsigned int);

 
///////////////////////////////////////////////////////////////////////////
    // declarations only
    template <typename Archive>
    void save(Archive& ar, hpx::geometry::polygon_2d const& ep, unsigned
int);

    template <typename Archive>
    void load(Archive& ar, hpx::geometry::polygon_2d& e, unsigned int);
}}

// load and save are separate functions
BOOST_SERIALIZATION_SPLIT_FREE(hpx::geometry::point);
BOOST_SERIALIZATION_SPLIT_FREE(hpx::geometry::polygon_2d);


