//  Copyright (c) 2017 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// For compliance with the NVIDIA EULA:
// "This software contains source code provided by NVIDIA Corporation."

// This is a conversion of the NVIDIA cublas example matrixMulCUBLAS to use
// HPX style data structures, executors and futures and demonstrate a simple use
// of computing a number of iteration of a matrix multiply on a stream and returning
// a future when it completes. This can be used to chain/schedule other task
// in a manner consistent with the future based API of HPX.
//
// Example usage: bin/cublas_matmul --sizemult=10 --iterations=25 --hpx:threads=8
// NB. The hpx::threads param only controls how many parallel tasks to use for the CPU
// comparison/checks and makes no difference to the GPU execution.
//
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_copy.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/parallel_executor_parameters.hpp>
//
#include <hpx/compute/cuda/target.hpp>
#include <hpx/compute/cuda/allocator.hpp>
#include <hpx/include/compute.hpp>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
//
#include <algorithm>

const char *_cudaGetErrorEnum(cublasStatus_t error);

// -------------------------------------------------------------------------
// a simple cublas wrapper helper object that can be used to synchronize
// cublas calls with an hpx future.
// -------------------------------------------------------------------------
template<typename T>
struct cublas_helper
{
public:
    using future_type    = hpx::future<void>;
    using allocator_type = typename hpx::compute::cuda::allocator<T>;
    using vector_type    = typename hpx::compute::vector<T, allocator_type>;

    // construct a cublas stream
    cublas_helper(std::size_t device=0) : target_(device) {
        handle_ = 0;
        stream_ = target_.native_handle().get_stream();
        return_ = cublasCreate(&handle_);
        cublas_error(return_);
    }

    ~cublas_helper() {
        return_ = cublasDestroy(handle_);
        cublas_error(return_);
    }

    // This is a simple wrapper for any cublas call, pass in the same arguments
    // that you would use for a cublas call except the cublas handle which is omitted
    // as the wrapper will supply that for you
    template <typename Func, typename ...Args>
    void cublas_wrapper(Func && cublas_function, Args&&... args)
    {
        // make sure this operation takes place on our stream
        return_ = cublasSetStream(handle_, stream_);
        cublas_error(return_);

        // insert the cublas handle in the arg list and call the cublas function
        return_ = cublas_function(handle_, std::forward<Args>(args)...);
        cublas_error(return_);
    }

    // get the future to synchronize this cublas stream with
    future_type get_future() { return target_.get_future(); }

    // return a copy of the cublas handle
    cublasHandle_t handle() { return handle_; }

    // return a reference to the compute::cuda object owned by this class
    hpx::compute::cuda::target & target() { return target_; }

    static void cublas_error(cublasStatus_t err) {
        if (err != CUBLAS_STATUS_SUCCESS) {
            std::stringstream temp;
            temp << "cublasDestroy returned error code " << _cudaGetErrorEnum(err);
            throw std::runtime_error(temp.str());
        }
    }

private:
    cublasHandle_t             handle_;
    cublasStatus_t             return_;
    cudaStream_t               stream_;
    hpx::compute::cuda::target target_;
};

// -------------------------------------------------------------------------
// Optional Command-line multiplier for matrix sizes
struct sMatrixSize {
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
};

// -------------------------------------------------------------------------
// Compute reference data set matrix multiply on CPU
// C = A * B
// @param C          reference data, computed but preallocated
// @param A          matrix A as provided to device
// @param B          matrix B as provided to device
// @param hA         height of matrix A
// @param wB         width of matrix B
// -------------------------------------------------------------------------
template <typename T>
void matrixMulCPU(T *C, const T *A, const T *B,
    unsigned int hA, unsigned int wA, unsigned int wB)
{
    hpx::parallel::for_loop(
        hpx::parallel::execution::par, 0, hA,
        [&](int i) {
            for (unsigned int j = 0; j < wB; ++j) {
                double sum = 0;
                for (unsigned int k = 0; k < wA; ++k) {
                    double a = A[i * wA + k];
                    double b = B[k * wB + j];
                    sum += a * b;
                }
                C[i * wB + j] = (T)sum;
            }
    });
}

// -------------------------------------------------------------------------
// Compute the L2 norm difference between two arrays
inline bool
compare_L2_err(const float *reference, const float *data,
               const unsigned int len, const float epsilon)
{
    assert(epsilon >= 0);

    float error = 0;
    float ref = 0;

    hpx::parallel::for_loop(
        hpx::parallel::execution::par, 0, len,
        [&](int i) {
            float diff = reference[i] - data[i];
            error += diff * diff;
            ref += reference[i] * reference[i];
        }
    );

    float normRef = sqrtf(ref);
    if (fabs(ref) < 1e-7) {
        return false;
    }

    float normError = sqrtf(error);
    error = normError / normRef;
    bool result = error < epsilon;
    return result;
}

// -------------------------------------------------------------------------
const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

// -------------------------------------------------------------------------
// Run a simple test matrix multiply using CUBLAS
// -------------------------------------------------------------------------
template <typename T>
void matrixMultiply(sMatrixSize &matrix_size, std::size_t device, std::size_t iterations)
{
    using hpx::parallel::execution::par;

    // Allocate host memory for matrices A and B
    unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;

    std::vector<T> h_A(size_A);
    std::vector<T> h_B(size_B);
    std::vector<T> h_C(size_C);
    std::vector<T> h_CUBLAS(size_C);

    // Fill A and B with random numbers
    auto randfunc = [](T &x) { x = rand() / (T)RAND_MAX; };
    hpx::parallel::for_each(par, h_A.begin(), h_A.end(), randfunc);
    hpx::parallel::for_each(par, h_B.begin(), h_B.end(), randfunc);

    // create a cublas helper object we'll use to futurize the cuda events
    using device_allocator = typename cublas_helper<T>::allocator_type;
    using device_vector    = typename cublas_helper<T>::vector_type;
    cublas_helper<T> cublas(device);

    // Create a cuda allocator
    device_allocator alloc(cublas.target());

    // Allocate device memory
    device_vector d_A(size_A, alloc);
    device_vector d_B(size_B, alloc);
    device_vector d_C(size_C, alloc);

    // The policy used in the parallel algorithms, just used default for now
    auto policy = hpx::parallel::execution::par;

    // copy host memory to device
    hpx::parallel::copy(policy, h_A.begin(), h_A.end(), d_A.begin());
    hpx::parallel::copy(policy, h_B.begin(), h_B.end(), d_B.begin());

    // create and start timer
    std::cout << "Computing result using CUBLAS...\n";
    // CUBLAS version 2.0
    const T alpha = 1.0f;
    const T beta  = 0.0f;
    //
    // Perform warmup operation with cublas
    // note cublas is column major ordering : transpose the order
    //
    hpx::util::high_resolution_timer t1;
    //
    cublas.cublas_wrapper(
        &cublasSgemm,
        CUBLAS_OP_N, CUBLAS_OP_N,
        matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA,
        &alpha,
        d_B.device_data(), matrix_size.uiWB,
        d_A.device_data(), matrix_size.uiWA,
        &beta,
        d_C.device_data(), matrix_size.uiWA);

    cublas.get_future().get();
    //            .then(
    //            [&t1](hpx::future<void> &&f) {
    double us1 = t1.elapsed_microseconds();
    std::cout << "warmup: elapsed_microseconds " << us1 << std::endl;
    //            }
    //        );

    // create a second stream for the main calculation
//    cublas_helper<T> cublas2(device);

    hpx::util::high_resolution_timer t2;
    for (int j = 0; j < iterations; j++)
    {
        cublas.cublas_wrapper(
            &cublasSgemm,
            CUBLAS_OP_N, CUBLAS_OP_N,
            matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA,
            &alpha,
            d_B.device_data(), matrix_size.uiWB,
            d_A.device_data(), matrix_size.uiWA,
            &beta,
            d_C.device_data(), matrix_size.uiWA);
    }

    cublas.get_future().get();
    double us2 = t2.elapsed_microseconds();
    std::cout << "actual: elapsed_microseconds " << us2
        << " iterations " << iterations << std::endl;

    // Compute and print the performance
    double usecPerMatrixMul = us2 / iterations;
    double flopsPerMatrixMul = 2.0 * (double)matrix_size.uiWA *
        (double)matrix_size.uiHA * (double)matrix_size.uiWB;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9) / (usecPerMatrixMul / 1e6);
    printf(
        "Performance = %.2f GFlop/s, Time = %.3f msec/iteration, Size = %.0f Ops\n",
        gigaFlops,
        1e-3*usecPerMatrixMul,
        flopsPerMatrixMul);

    // copy result from device to host
    hpx::parallel::copy(policy, d_C.begin(), d_C.end(), h_CUBLAS.begin());

    // compute reference solution on the CPU
    std::cout << "\nComputing result using host CPU...\n";
    // allocate storage for the CPU result
    std::vector<T> reference(size_C);

    hpx::util::high_resolution_timer t3;
    matrixMulCPU<T>(reference.data(), h_A.data(), h_B.data(),
        matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);
    double us3 = t3.elapsed_microseconds();
    std::cout << "CPU elapsed_microseconds (1 iteration) " << us3 << std::endl;

    // check result (CUBLAS)
    bool resCUBLAS = compare_L2_err(reference.data(), h_CUBLAS.data(), size_C, 1.0e-6f);
    if (resCUBLAS != true) {
        throw std::runtime_error("matrix CPU/GPU comparison error");
    }
    // if the result was incorrect, we throw an exception, so here it's ok
    std::cout << "\nComparing CUBLAS Matrix Multiply with CPU results: OK \n";
}

// -------------------------------------------------------------------------
int hpx_main(boost::program_options::variables_map& vm)
{
    std::size_t device     = vm["device"].as<std::size_t>();
    std::size_t sizeMult   = vm["sizemult"].as<std::size_t>();
    std::size_t iterations = vm["iterations"].as<std::size_t>();
    //
    unsigned int seed = (unsigned int)std::time(nullptr);
     if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);
    //
    sizeMult = (std::min)(sizeMult, std::size_t(100));
    sizeMult = (std::max)(sizeMult, std::size_t(1));
    //
    // use a larger block size for Fermi and above, query default cuda target properties
    hpx::compute::cuda::target target(device);

    std::cout
        << "GPU Device " << target.native_handle().get_device()
        << ": \"" << target.native_handle().processor_name() << "\" "
        << "with compute capability " << target.native_handle().processor_family()
        << "\n";

    int block_size = (target.native_handle().processor_family() < 2) ? 16 : 32;

    sMatrixSize matrix_size;
    matrix_size.uiWA = 2 * block_size * sizeMult;
    matrix_size.uiHA = 4 * block_size * sizeMult;
    matrix_size.uiWB = 2 * block_size * sizeMult;
    matrix_size.uiHB = 4 * block_size * sizeMult;
    matrix_size.uiWC = 2 * block_size * sizeMult;
    matrix_size.uiHC = 4 * block_size * sizeMult;

    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n\n",
           matrix_size.uiWA, matrix_size.uiHA,
           matrix_size.uiWB, matrix_size.uiHB,
           matrix_size.uiWC, matrix_size.uiHC);

    matrixMultiply<float>(matrix_size, device, iterations);
    return hpx::finalize();
}

// -------------------------------------------------------------------------
int main(int argc, char **argv)
{
    printf("[HPX Matrix Multiply CUBLAS] - Starting...\n");

    using namespace boost::program_options;
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");
    cmdline.add_options()
        (   "device",
            boost::program_options::value<std::size_t>()->default_value(0),
            "Device to use")
        (   "sizemult",
            boost::program_options::value<std::size_t>()->default_value(5),
            "Multiplier")
        (   "iterations",
            boost::program_options::value<std::size_t>()->default_value(30),
            "iterations")
        (   "no-cpu",
            boost::program_options::value<bool>()->default_value(false),
            "disable CPU validation to save time")
        ("seed,s",
            boost::program_options::value<unsigned int>(),
            "the random number generator seed to use for this run")
        ;

    return hpx::init(cmdline, argc, argv);
}
