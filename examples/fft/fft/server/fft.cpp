#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/serialization.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

#include <vector>
#include "fft.hpp"
namespace fft { namespace server
{    
    typedef fourier_xform::complex_type complex_type;
    typedef fourier_xform::complex_vector complex_vector;
    
    fourier_xform::fourier_xform(){};
    fourier_xform::~fourier_xform(){};
             
    complex_vector fourier_xform::sdft(complex_vector const& x)
    {
        //////////////////////////////////////////////////////////////////////////////
        complex_vector X;
        std::size_t n = x.size();                                                
        for(std::size_t k = 0; k < n; ++k)                                       
        {                                                                        
            std::size_t n1 = 0;                                                  
            fft::complex_type temp(0,0);                                         
            for(complex_vector::const_iterator itrb = x.begin();                 
                itrb != x.end(); ++itrb, ++n1)                                   
            {                                                                    
               temp = fourier_xform::complex_add(temp, fourier_xform::complex_mult
                   (*itrb, fourier_xform::complex_from_exponential(1, 
                    -2*pi*n1*k/n)));
            }                                                                    
            X.push_back(temp);                                                   
        }                                                                        
        return X;
    }

    complex_vector fourier_xform::r2ditfft(complex_vector const& x)
    {
        ///////////////////////////////////////////////////////////////////////
        complex_vector Y;                                                        
        std::size_t n = x.size();                                                
        
        if(n > 0)
        {
            if(n == 1)                                                               
            {                                                                        
                Y = x;                                                               
                return Y;                                                            
            }                                                                        
            complex_vector yeven, yodd, Yeven, Yodd;                                 
            complex_vector::const_iterator itr = x.begin();                          
                                                                                 
            std::size_t k = 0;                                                       
            while(k < n/2)                                                           
            {                                                                        
                yeven.push_back(*itr);  //yeven[k] = x[2*k];                         
                ++itr;                                                               
                yodd.push_back(*itr);   //yodd[k] = x[2*k + 1];                      
                ++itr;                                                               
                ++k;                                                                 
            }                                                                        
                                                                                 
            Yeven = r2ditfft(yeven);                                                 
            Yodd = r2ditfft(yodd);                                                   
            complex_vector::iterator itr1, itr2, itr_even, itr_odd;                  
            complex_vector Y_less_r, Y_gtr_r;                                        
                                                                                 
            itr_even = Yeven.begin();                                                
            itr_odd = Yodd.begin();                                                  
                                                                                 
            for (k = 0; k < n/2; ++k){                                               
                Y_less_r.push_back(fourier_xform::complex_add(*itr_even
                    , fourier_xform::complex_mult(*itr_odd
                    , fourier_xform::complex_from_exponential(1, -2*pi*k/n))));                
            
                Y_gtr_r.push_back(fourier_xform::complex_sub(*itr_even
                    , fourier_xform::complex_mult(*itr_odd       
                    , fourier_xform::complex_from_exponential(1, -2*pi*k/n))));
                ++itr_even;                                                          
                ++itr_odd;                                                           
            }                                                                        
                                                                                 
            for(itr1 = Y_less_r.begin(); itr1 != Y_less_r.end(); ++itr1)             
            {                                                                        
                Y.push_back(*itr1);                                                  
            }                                                                        
                                                                                 
            for(itr1 = Y_gtr_r.begin(); itr1 != Y_gtr_r.end(); ++itr1)               
            {                                                                        
                Y.push_back(*itr1);                                                  
            }                                                                        
            return Y;
        }
        else
        {
            ///Throw HPX error. 
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "fft::fourier_xform::r2ditfft",
                "value of parameter 'what' is not valid");
            
        }
    }

    complex_vector fourier_xform::r2ditfft_args(complex_vector const& y_even
        , complex_vector const& y_odd)
    {
        //x1 is always even
        //x2 is always odd
        ///////////////////////////////////////////////////////////////////////
        complex_vector Y;                                                        
        std::size_t n = y_even.size();
        BOOST_ASSERT(n == y_odd.size());
        
        complex_vector::iterator itr1, itr2;//, itr_even, itr_odd;                  
        complex_vector Y_less_r, Y_gtr_r;                                        

        complex_vector::const_iterator itr_even = y_even.begin();                                                
        complex_vector::const_iterator itr_odd = y_odd.begin();                                                  


        // FIX ME: 
        for (std::size_t k = 0; k < n; ++k){                                               
            Y_less_r.push_back(fourier_xform::complex_add(*itr_even
                , fourier_xform::complex_mult(*itr_odd
                , fourier_xform::complex_from_exponential(1, -pi*k/n))));                

            Y_gtr_r.push_back(fourier_xform::complex_sub(*itr_even
                , fourier_xform::complex_mult(*itr_odd       
                , fourier_xform::complex_from_exponential(1, -pi*k/n))));
            ++itr_even;                                                          
            ++itr_odd;                                                           
        }                                                                        

        for(itr1 = Y_less_r.begin(); itr1 != Y_less_r.end(); ++itr1)             
        {                                                                        
            Y.push_back(*itr1);                                                  
        }                                                                        

        for(itr1 = Y_gtr_r.begin(); itr1 != Y_gtr_r.end(); ++itr1)               
        {                                                                        
            Y.push_back(*itr1);                                                  
        }                                                                        
        return Y; 
    }
    ///////////////////////////////////////////////////////////////////////////
    
    double fourier_xform::magnitude_from_complex(complex_type const& temp)
    {
        return sqrt(temp.re * temp.re + temp.im * temp.im);
    }

    complex_type fourier_xform::complex_from_exponential(double const& mag, 
        double const& theta_rad)
    {
        complex_type y;
        y.re = mag * cos(theta_rad);
        if(y.re == -0)
            y.re = 0;

        y.im = mag * sin(theta_rad);
        if(y.im == -0)
            y.im = 0;

        return y;
    }

    complex_type fourier_xform::complex_mult(complex_type const& a
        , complex_type const& b)
    {
        complex_type y;
        y.re = (a.re * b.re) - (a.im * b.im);
        if(y.re == -0)
            y.re = 0;

        y.im = (a.re * b.im) + (a.im * b.re);
        if(y.im == -0)
            y.im = 0;

        return y; 
    }

    complex_type fourier_xform::complex_add(complex_type const& a
        , complex_type const& b)
    {
        complex_type y;
        y.re = a.re + b.re;
        if(y.re == -0)
            y.re = 0;

        y.im = a.im + b.im;
        if(y.im == -0)
            y.im = 0;

        return y;
    }

    complex_type fourier_xform::complex_sub(complex_type const& a
        , complex_type const& b)
    {
        complex_type y;
        y.re = a.re - b.re;
        if(y.re == -0)
            y.re = 0;

        y.im = a.im - b.im;
        if(y.im == -0)
            y.im = 0;

        return y;
    }

}}
