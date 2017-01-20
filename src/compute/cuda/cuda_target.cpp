//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CUDA)

#include <hpx/exception.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/runtime/find_here.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/assert.hpp>

#include <hpx/compute/cuda/target.hpp>

#include <cstddef>
#include <string>
#include <utility>

#include <cuda_runtime.h>

#include <boost/intrusive_ptr.hpp>

namespace hpx { namespace compute { namespace cuda
{
    namespace detail
    {
        struct runtime_registration_wrapper
        {
            runtime_registration_wrapper(hpx::runtime* rt)
              : rt_(rt)
            {
                HPX_ASSERT(rt);

                // Register this thread with HPX, this should be done once for
                // each external OS-thread intended to invoke HPX functionality.
                // Calling this function more than once on the same thread will
                // report an error.
                hpx::error_code ec(hpx::lightweight);       // ignore errors
                hpx::register_thread(rt_, "cuda", ec);
            }
            ~runtime_registration_wrapper()
            {
                // Unregister the thread from HPX, this should be done once in
                // the end before the external thread exists.
                hpx::unregister_thread(rt_);
            }

            hpx::runtime* rt_;
        };

        ///////////////////////////////////////////////////////////////////////
        struct future_data : lcos::detail::future_data<void>
        {
        private:
            static void CUDART_CB stream_callback(cudaStream_t stream,
                cudaError_t error, void* user_data);

        public:
            future_data();

            void init(cudaStream_t stream);

        private:
            hpx::runtime* rt_;
        };

        struct release_on_exit
        {
            release_on_exit(future_data* data)
              : data_(data)
            {}

            ~release_on_exit()
            {
                // release the shared state
                lcos::detail::intrusive_ptr_release(data_);
            }

            future_data* data_;
        };

        ///////////////////////////////////////////////////////////////////////
        void CUDART_CB future_data::stream_callback(cudaStream_t stream,
            cudaError_t error, void* user_data)
        {
            future_data* this_ = static_cast<future_data*>(user_data);

            runtime_registration_wrapper wrap(this_->rt_);

            // We need to run this as an HPX thread ...
            hpx::applier::register_thread_nullary(
                [this_, error] ()
                {
                    release_on_exit on_exit(this_);

                    if (error != cudaSuccess)
                    {
                        this_->set_exception(
                            HPX_GET_EXCEPTION(kernel_error,
                                "cuda::detail::future_data::stream_callback()",
                                std::string("cudaStreamAddCallback failed: ") +
                                    cudaGetErrorString(error))
                        );
                        return;
                    }

                    this_->set_data(hpx::util::unused);
                },
                "hpx::compute::cuda::future_data::stream_callback"
            );
        }

        future_data::future_data()
          : rt_(hpx::get_runtime_ptr())
        {}

        void future_data::init(cudaStream_t stream)
        {
            // Hold on to the shared state on behalf of the cuda runtime
            // right away as the callback could be called immediately.
            lcos::detail::intrusive_ptr_add_ref(this);

            cudaError_t error = cudaStreamAddCallback(
                stream, stream_callback, this, 0);
            if (error != cudaSuccess)
            {
                // callback was not called, release object
                lcos::detail::intrusive_ptr_release(this);

                // report error
                HPX_THROW_EXCEPTION(kernel_error,
                    "cuda::detail::future_data::future_data()",
                    std::string("cudaStreamAddCallback failed: ") +
                        cudaGetErrorString(error));
            }
        }
    }

    void target::native_handle_type::init_processing_units()
    {
        cudaDeviceProp props;
        cudaError_t error = cudaGetDeviceProperties(&props, device_);
        if (error != cudaSuccess)
        {
            // report error
            HPX_THROW_EXCEPTION(kernel_error,
                "cuda::default_executor::processing_units_count()",
                std::string("cudaGetDeviceProperties failed: ") +
                    cudaGetErrorString(error));
        }

        std::size_t mp = props.multiProcessorCount;
        switch(props.major)
        {
            case 2: // Fermi
                if (props.minor == 1) {
                    mp = mp * 48;
                }
                else {
                    mp = mp * 32;
                }
                break;
            case 3: // Kepler
                mp = mp * 192;
                break;
            case 5: // Maxwell
                mp = mp * 128;
                break;
            case 6: // Pascal
                mp = mp * 64;
                break;
             default:
                break;
        }
        processing_units_ = mp;
        processor_family_ = props.major;
        processor_name_   = props.name;
    }

    target::native_handle_type::native_handle_type(int device)
      : device_(device), stream_(0)
    {
        init_processing_units();
    }

    target::native_handle_type::~native_handle_type()
    {
        reset();
    }

    void target::native_handle_type::reset() HPX_NOEXCEPT
    {
        if (stream_)
            cudaStreamDestroy(stream_);     // ignore error
    }

    target::native_handle_type::native_handle_type(
            target::native_handle_type const& rhs) HPX_NOEXCEPT
      : device_(rhs.device_),
        processing_units_(rhs.processing_units_),
        processor_family_(rhs.processor_family_),
        processor_name_(rhs.processor_name_),
        stream_(0)
    {
    }

    target::native_handle_type::native_handle_type(
            target::native_handle_type && rhs) HPX_NOEXCEPT
      : device_(rhs.device_),
        processing_units_(rhs.processing_units_),
        processor_family_(rhs.processor_family_),
        processor_name_(rhs.processor_name_),
        stream_(rhs.stream_)
    {
        rhs.stream_ = 0;
    }

    target::native_handle_type& target::native_handle_type::operator=(
        target::native_handle_type const& rhs) HPX_NOEXCEPT
    {
        if (this == &rhs)
            return *this;

        device_           = rhs.device_;
        processing_units_ = rhs.processing_units_;
        processor_family_ = rhs.processor_family_;
        processor_name_   = rhs.processor_name_;
        reset();

        return *this;
    }

    target::native_handle_type& target::native_handle_type::operator=(
        target::native_handle_type && rhs) HPX_NOEXCEPT
    {
        if (this == &rhs)
            return *this;

        device_           = rhs.device_;
        processing_units_ = rhs.processing_units_;
        processor_family_ = rhs.processor_family_;
        processor_name_   = rhs.processor_name_;
        stream_ = rhs.stream_;
        rhs.stream_ = 0;

        return *this;
    }

    cudaStream_t target::native_handle_type::get_stream() const
    {
        std::lock_guard<mutex_type> l(mtx_);

        if (stream_ == 0)
        {
            cudaError_t error = cudaSetDevice(device_);
            if (error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "cuda::target::native_handle::get_stream()",
                    std::string("cudaSetDevice failed: ") +
                        cudaGetErrorString(error));
            }
            error = cudaStreamCreateWithFlags(&stream_,
                cudaStreamNonBlocking);
            if (error != cudaSuccess)
            {
                HPX_THROW_EXCEPTION(kernel_error,
                    "cuda::target::native_handle::get_stream()",
                    std::string("cudaStreamCreate failed: ") +
                        cudaGetErrorString(error));
            }
        }
        return stream_;
    }

    ///////////////////////////////////////////////////////////////////////////
    void target::synchronize() const
    {
        // FIXME: implement remote targets
        HPX_ASSERT(hpx::find_here() == locality_);

        cudaStream_t stream = handle_.get_stream();

        if (stream == 0)
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "cuda::target::synchronize",
                "no stream available");
        }

        cudaError_t error = cudaStreamSynchronize(stream);
        if(error != cudaSuccess)
        {
            HPX_THROW_EXCEPTION(kernel_error,
                "cuda::target::synchronize",
                std::string("cudaStreamSynchronize failed: ") +
                    cudaGetErrorString(error));
        }
    }

    hpx::future<void> target::get_future() const
    {
        typedef detail::future_data shared_state_type;

        // make sure shared state stays alive even if the callback is invoked
        // during initialization
        boost::intrusive_ptr<shared_state_type> p(new shared_state_type());
        p->init(handle_.get_stream());
        return hpx::traits::future_access<hpx::future<void> >::
            create(std::move(p));
    }

    ///////////////////////////////////////////////////////////////////////////
    target& get_default_target()
    {
        static target target_;
        return target_;
    }
}}}

#endif

