//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SHENEOS_PARTITION3D_AUG_08_2011_1220PM)
#define HPX_SHENEOS_PARTITION3D_AUG_08_2011_1220PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/components/server/simple_component_base.hpp>

#include "../dimension.hpp"

namespace sheneos { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT partition3d
      : public hpx::components::simple_component_base<partition3d>
    {
    private:
        typedef hpx::components::simple_component_base<partition3d> base_type;

        inline void init_dimension(std::string const&, int, dimension const&, 
            char const*, boost::scoped_array<double>&);
        inline void init_data(std::string const& datafilename, 
            char const* name, boost::scoped_array<double>& values, 
            std::size_t array_size);
        inline std::size_t get_index(int d, double value);
        inline double interpolate(double* values, 
            std::size_t idx_x, std::size_t idx_y, std::size_t idx_z,
            double delta_ye, double delta_logtemp, double delta_logrho);

    public:
        enum eos_values {
            logpress = 0x00000001,  // pressure
            logenergy = 0x00000002, // specific internal energy
            entropy = 0x00000004,   // specific entropy
            munu = 0x00000008,      // mu_e - mun + mu_p
            cs2 = 0x00000010,       // speed of sound squared
        // derivatives
            dedt = 0x00000020,      // C_v
            dpdrhoe = 0x00000040,   // dp/deps|rho
            dpderho = 0x00000080,   // dp/drho|eps
            small_api_values = 0x000000FF,
#if SHENEOS_SUPPORT_FULL_API
        // chemical potentials
            muhat = 0x00000100,     // mu_n - mu_p
            mu_e = 0x00000200,      // electron chemical potential including electron rest mass
            mu_p = 0x00000400,      // proton chemical potential
            mu_n = 0x00000800,      // neutron chemical potential
        // compositions
            xa = 0x00001000,        // alpha particle number fraction
            xh = 0x00002000,        // heavy nucleus number fraction
            xn = 0x00004000,        // neutron number fraction
            xp = 0x00008000,        // proton number fraction
            abar = 0x00010000,      // average heavy nucleus A
            zbar = 0x00020000,      // average heavy nucleus Z
            gamma = 0x00040000,     // Gamma_1
            full_api_values = 0x0007FFFF,
#endif
        };

        partition3d();

        // components must contain a typedef for wrapping_type defining the
        // component type used to encapsulate instances of this component
        typedef partition3d wrapping_type;

        ///////////////////////////////////////////////////////////////////////
        // parcel action code: the action to be performed on the destination 
        // object (the accumulator)
        enum actions
        {
            partition3d_init = 0,
            partition3d_interpolate = 1
        };

        // exposed functionality
        void init(std::string const&, dimension const&, dimension const&, 
            dimension const&);
        std::vector<double> interpolate(double ye, double temp, double rho, 
            boost::uint32_t eosvalues);

        ///////////////////////////////////////////////////////////////////////
        // Each of the exposed functions needs to be encapsulated into an action
        // type, allowing to generate all required boilerplate code for threads,
        // serialization, etc.
        typedef hpx::actions::action4<
            partition3d, partition3d_init, std::string const&, 
            dimension const&, dimension const&, dimension const&,
            &partition3d::init
        > init_action;

        typedef hpx::actions::result_action4<
            partition3d, std::vector<double>, partition3d_interpolate, 
            double, double, double, boost::uint32_t, &partition3d::interpolate
        > interpolate_action;

    private:
        dimension dim_[dimension::dim];

        double min_value_[dimension::dim];
        double max_value_[dimension::dim];
        double delta_[dimension::dim];

        // values of independent variables
        boost::scoped_array<double> ye_values_;
        boost::scoped_array<double> logtemp_values_;
        boost::scoped_array<double> logrho_values_;

        double energy_shift_;

        // dependent variables
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

#endif



