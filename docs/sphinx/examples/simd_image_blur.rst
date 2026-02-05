..
    Copyright (C) 2025 Dimitra Karatza

    SPDX-License-Identifier: BSL-1.0
    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

.. _examples_image_blur:

==============================
Parallel SIMD image processing
==============================

Image processing is a classic example of a data-parallel workload: the same
operation is applied independently to many pixels. This makes it a natural fit
for SIMD vectorization and parallel execution.

This example demonstrates how to use |hpx| parallel algorithms together with
SIMD-enabled execution policies to apply a simple vertical blur filter to an
image. The example combines **task-level parallelism** and **data-level
parallelism** using the :cpp:var:`hpx::execution::par_simd` execution policy.

.. _simd_blur_setup:

Setup
=====
Setup
=====

The source code for this example can be found here:
:download:`simd_image_blur.cpp <../../examples/quickstart/simd_image_blur.cpp>`.

SIMD support must be enabled in your |hpx| build. Make sure |hpx| was configured with
the Data Parallelism module enabled, for example:

.. code-block:: shell-session

   -DHPX_WITH_DATAPAR=On
   -DHPX_WITH_DATAPAR_BACKEND=STD_EXPERIMENTAL_SIMD

To compile this program, go to your |hpx| build directory (see
:ref:`building_hpx` for information on configuring and building |hpx|) and enter:

.. code-block:: shell-session

   $ make examples.quickstart.simd_image_blur

To run the program, type:

.. code-block:: shell-session

   $ ./bin/simd_image_blur <input.bmp> <output.bmp>

where `<input.bmp>` is the BMP image you want to blur, and `<output.bmp>` is the
filename for the blurred result.

This example uses the EasyBMP library to load and store images. EasyBMP is a
lightweight BMP image library that provides direct access to pixel data. After execution,
the program will produce the blurred output image at the filename you specified.

.. _simd_blur_overview:

Overview of the blur operation
==============================

The blur filter implemented in this example is a simple vertical 3-point
stencil. For each pixel, a weighted average of the pixel itself and its
vertical neighbors is computed:

.. math::

   p_{out} = 0.25 \cdot p_{top} + 0.5 \cdot p_{center} + 0.25 \cdot p_{bottom}

This operation is applied independently to each color channel (red, green, and
blue). Border pixels are excluded from the computation to avoid out-of-bounds
memory accesses and are copied directly to the output image.

Explanation
===========

The first lines pull in the necessary |hpx| headers, the EasyBMP library, and
headers uch as vector, iosteam and algorithms.

Main function
-------------

The ``main`` function, which is responsible for loading the input image, preparing
the output image, invoking the blur kernel, and writing the result.

First, an input image object is created and populated from a file:

.. code-block:: c++

    BMP input;
    if (!input.ReadFromFile(argv[1])) {
        std::cerr << "Could not open " << argv[1] << "\n";
        return 1;
    }

The call to ``ReadFromFile()`` attempts to load the image. If the file
cannot be opened or read, the program prints an error message and exits.

Next, the dimensions of the image are queried:

.. code-block:: c++

    int width = input.TellWidth();
    int height = input.TellHeight();

These values are used to allocate an output image of the same size. The output
image is configured to use a 24-bit RGB format:

.. code-block:: c++

    BMP output;
    output.SetSize(width, height);
    output.SetBitDepth(24); // RGB

Before applying the blur, the input image is copied into the output image:

.. code-block:: c++

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            *output(x, y) = *input(x, y);

This step preserves the border pixels, which are not modified by the blur
operation. Copying the image up front avoids special-case handling for the
image boundaries inside the blur kernel.

Finally, the blur kernel is invoked and the resulting image is written:

.. code-block:: c++

    blur_image(input, output);

    output.WriteToFile("blurred.bmp");

At this point, the output image contains the blurred result, which is saved as
a new image file.

Blur function
-------------

The blur computation is implemented in the ``blur_image`` function. The input
image is passed as a read-only reference, while the output image is modified in
place:

.. code-block:: cpp

    void blur_image(const easy_bmp::BMP& input, easy_bmp::BMP& output)

The function begins by querying the dimensions of the image:

.. code-block:: cpp

    int height = input.TellHeight();

These values are used to construct the iteration space for the blur operation.

Defining the iteration space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|hpx| parallel algorithms operate over ranges. In this example, the blur is
applied to all rows of the image. A vector of row indices is constructed as
follows:

.. code-block:: cpp

    std::vector<int> rows(height);
    for (int i = 0; i < height; ++i)
        rows[i] = i;

Each element of the ``rows`` vector corresponds to a valid row index in the
range ``[0, height - 1]``. This guarantees that every processed row has valid
indices. The ``blur_row`` function itself checks column bounds to prevent
out-of-bounds access.

Parallel blur using ``par_simd``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The blur computation is expressed using :cpp:func:`hpx::for_each` with the
:cpp:var:`hpx::execution::par_simd` execution policy. The lambda is generic
to support both SIMD and scalar calls:

.. code-block:: cpp

    hpx::for_each(
        hpx::execution::par_simd,
        rows.begin(), rows.end(),
        [&](auto y_vec) {
            if constexpr (hpx::parallel::traits::is_vector_pack_v<decltype(y_vec)>) {
                for (size_t i = 0; i < y_vec.size(); ++i) {
                    int y = y_vec[i];
                    blur_row(input, output, y);
                }
            } else {
                int y = y_vec;
                blur_row(input, output, y);
            }
        }
    );

The ``par_simd`` execution policy enables HPX to distribute iterations across
multiple worker threads while also applying SIMD vectorization within each
thread. Each iteration of the loop processes one or more rows, depending on
whether a SIMD pack or scalar value is passed.

Processing pixels within a row
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inside the ``blur_row`` function, the code iterates over all columns of the
current row:

.. code-block:: cpp

    for (int x = 0; x < width; ++x) {

For each pixel at position ``(x, y)``, neighboring pixels in a 3x3 kernel are
accessed to compute the blur:

.. code-block:: cpp

    for (int ky = -1; ky <= 1; ++ky) {
        int ny = y + ky;
        if (ny < 0 || ny >= height) continue;
        for (int kx = -1; kx <= 1; ++kx) {
            int nx = x + kx;
            if (nx < 0 || nx >= width) continue;
            auto pixel = input(nx, ny);
            r += pixel->Red;
            g += pixel->Green;
            b += pixel->Blue;
            ++count;
        }
    }

Blurring individual color channels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The blur operation is applied independently to each color channel (red, green,
and blue). After summing the contributions from the kernel, the average is
computed and assigned to the output pixel:

.. code-block:: cpp

    output(x, y)->Red   = r / count;
    output(x, y)->Green = g / count;
    output(x, y)->Blue  = b / count;

This ensures that each output pixel represents the average color of its
neighbors, implementing a simple 3x3 blur.


Combining task-level and data-level parallelism
-----------------------------------------------

By parallelizing over image rows and using the ``par_simd`` execution policy,
this example combines two forms of parallelism:

* **Task-level parallelism**, by processing different rows concurrently
* **Data-level parallelism (SIMD)**, by vectorizing the computations within each
  thread

This allows the blur operation to scale efficiently on modern multicore CPUs
with SIMD support, without requiring explicit SIMD intrinsics or manual thread
management.

Summary
=======

This example illustrates how HPX execution policies enable expressive and
efficient SIMD-aware parallel programming. Real-world workloads such as image
processing can be parallelized with minimal effort, while still allowing fine-
grained control over execution behavior through policy selection.
