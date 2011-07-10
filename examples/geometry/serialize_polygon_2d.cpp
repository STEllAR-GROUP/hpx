
#include <hpx/hpx_fwd.hpp> 
#include "serialize_polygon_2d.hpp"

namespace hpx { namespace geometry
{
 
///////////////////////////////////////////////////////////////////////////
    // implement the serialization functions
    template <typename Archive>
    void save(Archive& ar, hpx::geometry::point const& p, unsigned int)
    {
        hpx::naming::id_type id = p.get_gid();
        ar & id;
    }

    template <typename Archive>
    void load(Archive& ar, hpx::geometry::point& p, unsigned int)
    {
        hpx::naming::id_type id;
        ar & id;
        p = hpx::geomtr::point(id);
    }

    template <typename Archive>
    void save(Archive& ar, hpx::geometry::polygon_2d const& p, unsigned int)
    {
        ar & p.outer();
        // no need to serialize the inner rings for now
    }

    template <typename Archive>
    void load(Archive& ar, hpx::geometry::polygon_2d& p, unsigned int)
    {
        ar & p.outer();
        // no need to serialize the inner rings for now
    }

///////////////////////////////////////////////////////////////////////////
    // explicit instantiation for the correct archive types
#if HPX_USE_PORTABLE_ARCHIVES != 0
    template HPX_EXPORT void 
    save(hpx::util::portable_binary_oarchive&, hpx::geometry::polygon_2d
const&, 
        unsigned int);

    template HPX_EXPORT void 
    load(hpx::util::portable_binary_iarchive&, hpx::geometry::polygon_2d&, 
        unsigned int);
#else
    template HPX_EXPORT void 
    save(boost::archive::binary_oarchive&, hpx::geometry::polygon_2d const&,

        unsigned int);

    template HPX_EXPORT void 
    load(boost::archive::binary_iarchive&, hpx::geometry::polygon_2d&, 
        unsigned int);
#endif
}}

