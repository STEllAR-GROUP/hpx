//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SHENEOS_PARTITION3D_AUG_08_2011_1220PM)
#define HPX_SHENEOS_PARTITION3D_AUG_08_2011_1220PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>
#include <hpx/runtime/serialization/serialize.hpp>

#include "../dimension.hpp"

namespace sheneos
{
    ///////////////////////////////////////////////////////////////////////////
    struct sheneos_coord
    {
        sheneos_coord(double ye = 0.0, double temp = 0.0, double rho = 0.0)
          : ye_(ye), temp_(temp), rho_(rho)
        {}

        double ye_;
        double temp_;
        double rho_;
    };
}

namespace sheneos { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT partition3d
      : public hpx::components::simple_component_base<partition3d>
    {
        inline void init_dimension(std::string const&, int, dimension const&,
            char const*, boost::scoped_array<double>&);

        inline void init_data(std::string const& datafilename,
            char const* name, boost::scoped_array<double>& values,
            std::size_t array_size);

        /// Get index of a given value in the one-dimensional array-slice.
        inline std::size_t get_index(dimension::type d, double value);

        /// Tri-linear interpolation routine.
        inline double tl_interpolate(double* values,
            std::size_t idx_x, std::size_t idx_y, std::size_t idx_z,
            double delta_ye, double delta_logtemp, double delta_logrho);

    public:
        enum eos_values {
            logpress = 0x00000001,  ///< pressure
            logenergy = 0x00000002, ///< specific internal energy
            entropy = 0x00000004,   ///< specific entropy
            munu = 0x00000008,      ///< mu_e - mun + mu_p
            cs2 = 0x00000010,       ///< speed of sound squared
            // Derivatives.
            dedt = 0x00000020,      ///< C_v
            dpdrhoe = 0x00000040,   ///< dp/deps|rho
            dpderho = 0x00000080,   ///< dp/drho|eps
            small_api_values = 0x000000FF,
#if SHENEOS_SUPPORT_FULL_API
            // Chemical potentials.
            muhat = 0x00000100,     ///< mu_n - mu_p
            mu_e = 0x00000200,      ///< electron chemical potential
                                    ///< including electron rest mass
            mu_p = 0x00000400,      ///< proton chemical potential
            mu_n = 0x00000800,      ///< neutron chemical potential
            // Compositions.
            xa = 0x00001000,        ///< alpha particle number fraction
            xh = 0x00002000,        ///< heavy nucleus number fraction
            xn = 0x00004000,        ///< neutron number fraction
            xp = 0x00008000,        ///< proton number fraction
            abar = 0x00010000,      ///< average heavy nucleus A
            zbar = 0x00020000,      ///< average heavy nucleus Z
            gamma = 0x00040000,     ///< Gamma_1
            full_api_values = 0x0007FFFF,
#endif
        };

        ///////////////////////////////////////////////////////////////////////
        partition3d();

        ///////////////////////////////////////////////////////////////////////
        // Exposed functionality.

        /// Initialize this partition.
        void init(std::string const& datafilename, dimension const& dimx,
            dimension const& dimy, dimension const& dimz);

        /// Perform an interpolation on this partition.
        ///
        /// \param ye        [in] Electron fraction.
        /// \param temp      [in] Temperature.
        /// \param rho       [in] Rest mass density of the plasma.
        /// \param eosvalues [in] The EOS values to interpolate. Must be
        ///                  in the range of this partition.
        std::vector<double> interpolate(double ye, double temp, double rho,
            boost::uint32_t eosvalues);

        /// Perform an interpolation of one given field on this partition.
        ///
        /// \param ye        [in] Electron fraction.
        /// \param temp      [in] Temperature.
        /// \param rho       [in] Rest mass density of the plasma.
        /// \param eosvalues [in] The EOS value to interpolate. Must be
        ///                  in the range of this partition.
        double interpolate_one(double ye, double temp, double rho,
            boost::uint32_t eosvalue);

        /// Perform several interpolations of all given fields on this partition.
        ///
        /// \param cords     [in] triples of electron fractions, temperatures,
        ///                  and rest mass densities of the plasma.
        /// \param eosvalues [in] The EOS values to interpolate. Must be
        ///                  in the range of this partition.
        std::vector<std::vector<double> >
        interpolate_bulk(std::vector<sheneos_coord> const& coords,
            boost::uint32_t eosvalues);

        /// Perform several interpolations of one given field on this partition.
        ///
        /// \param cords     [in] triples of electron fractions, temperatures,
        ///                  and rest mass densities of the plasma.
        /// \param eosvalue  [in] The EOS value to interpolate. Must be
        ///                  in the range of this partition.
        std::vector<double>
        interpolate_one_bulk(std::vector<sheneos_coord> const& coords,
            boost::uint32_t eosvalue);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an
        // action type, generating all required boilerplate code for threads,
        // serialization, etc.
        HPX_DEFINE_COMPONENT_ACTION(partition3d, init);
        HPX_DEFINE_COMPONENT_ACTION(partition3d, interpolate);
        HPX_DEFINE_COMPONENT_ACTION(partition3d, interpolate_one);
        HPX_DEFINE_COMPONENT_ACTION(partition3d, interpolate_bulk);
        HPX_DEFINE_COMPONENT_ACTION(partition3d, interpolate_one_bulk);

    protected:
        double interpolate_one(sheneos_coord const& c,
            boost::uint32_t eosvalue);
        std::vector<double> interpolate(sheneos_coord const& c,
            boost::uint32_t eosvalues);

    private:
        dimension dim_[dimension::dim];

        double min_value_[dimension::dim];
        double max_value_[dimension::dim];
        double delta_[dimension::dim];

        // Values of independent variables.
        boost::scoped_array<double> ye_values_;
        boost::scoped_array<double> logtemp_values_;
        boost::scoped_array<double> logrho_values_;

        double energy_shift_;

        // Dependent variables.
        boost::scoped_array<double> logpress_values_;
        boost::scoped_array<double> logenergy_values_;
        boost::scoped_array<double> entropy_values_;
        boost::scoped_array<double> munu_values_;
        boost::scoped_array<double> cs2_values_;
        boost::scoped_array<double> dedt_values_;
        boost::scoped_array<double> dpdrhoe_values_;
        boost::scoped_array<double> dpderho_values_;
#if SHENEOS_SUPPORT_FULL_API
        boost::scoped_array<double> muhat_values_;
        boost::scoped_array<double> mu_e_values_;
        boost::scoped_array<double> mu_p_values_;
        boost::scoped_array<double> mu_n_values_;
        boost::scoped_array<double> xa_values_;
        boost::scoped_array<double> xh_values_;
        boost::scoped_array<double> xn_values_;
        boost::scoped_array<double> xp_values_;
        boost::scoped_array<double> abar_values_;
        boost::scoped_array<double> zbar_values_;
        boost::scoped_array<double> gamma_values_;
#endif
    };
}}

///////////////////////////////////////////////////////////////////////////////
// Non-intrusive serialization.
namespace hpx { namespace serialization
{
    HPX_COMPONENT_EXPORT void
    serialize(input_archive& ar,
        sheneos::sheneos_coord& coord, unsigned int const);

    HPX_COMPONENT_EXPORT void
    serialize(output_archive& ar,
        sheneos::sheneos_coord& coord, unsigned int const);
}}

///////////////////////////////////////////////////////////////////////////////
HPX_REGISTER_ACTION_DECLARATION(
    sheneos::server::partition3d::init_action,
    sheneos_partition3d_init_action);
HPX_ACTION_USES_LARGE_STACK(sheneos::server::partition3d::init_action);

HPX_REGISTER_ACTION_DECLARATION(
    sheneos::server::partition3d::interpolate_action,
    sheneos_partition3d_interpolate_action);
HPX_REGISTER_ACTION_DECLARATION(
    sheneos::server::partition3d::interpolate_one_action,
    sheneos_partition3d_interpolate_one_action);

HPX_REGISTER_ACTION_DECLARATION(
    sheneos::server::partition3d::interpolate_bulk_action,
    sheneos_partition3d_interpolate_bulk_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::base_lco_with_value<std::vector<std::vector<double> > >::set_value_action,
    set_value_action_vector_vector_double);

HPX_REGISTER_ACTION_DECLARATION(
    sheneos::server::partition3d::interpolate_one_bulk_action,
    sheneos_partition3d_interpolate_one_bulk_action);
HPX_REGISTER_ACTION_DECLARATION(
    hpx::lcos::base_lco_with_value<std::vector<double> >::set_value_action,
    set_value_action_vector_double);

#endif

