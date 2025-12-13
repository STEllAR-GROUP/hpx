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

SIMD support must be enabled in your |hpx| build. Make sure |hpx| was configured with
the Data Parallelism module enabled, for example:

.. code-block:: shell-session

   -DHPX_WITH_DATAPAR=On
   -DHPX_WITH_DATAPAR_BACKEND=STD_EXPERIMENTAL_SIMD

This example uses the EasyBMP library to load and store images. EasyBMP is a
lightweight BMP image library that provides direct access to pixel data.

Users are free to use any image library and any image they want to blur,
as long as the library allows reading pixel values and writing the processed
output back to an image file.

The program reads an input image, applies a simple vertical blur filter, and
writes the blurred result to an output image. Only the blur computation itself
depends on |hpx|.

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

Full example code
=================

The following listing shows the complete example. The image handling code may
vary depending on the chosen image library, but the blur kernel and |hpx| usage
remain the same.

.. code-block:: c++

    #include <hpx/hpx_main.hpp>
    #include <hpx/include/parallel_algorithm.hpp>
    #include "./include/EasyBMP.h"
    #include <vector>
    #include <iostream>
    #include <algorithm>

    using namespace easy_bmp;

    // Apply vertical 3-point blur to each RGB channel separately
    void blur_image(const BMP& input, BMP& output) {
        int width = input.TellWidth();
        int height = input.TellHeight();

        // Prepare rows to parallelize over (excluding borders)
        std::vector<int> rows(height - 2);
        for (int i = 0; i < height - 2; ++i)
            rows[i] = i + 1;

        // Perform the blur
        hpx::for_each(
            hpx::execution::par_simd,
            rows.begin(), rows.end(),
            [&](int y) {
                for (int x = 0; x < width; ++x) {
                    const RGBApixel* top = input(x, y - 1);
                    const RGBApixel* mid = input(x, y);
                    const RGBApixel* bot = input(x, y + 1);

                    auto blur_channel = [](int a, int b, int c) {
                        return std::clamp(static_cast<int>(0.25f * a + 0.5f * b + 0.25f * c), 0, 255);
                    };

                    output(x, y)->Red   = blur_channel(top->Red,   mid->Red,   bot->Red);
                    output(x, y)->Green = blur_channel(top->Green, mid->Green, bot->Green);
                    output(x, y)->Blue  = blur_channel(top->Blue,  mid->Blue,  bot->Blue);
                }
            }
        );
    }

    int main() {
        BMP input;
        if (!input.ReadFromFile("image.bmp")) {
            std::cerr << "Could not open image.bmp\n";
            return 1;
        }

        int width = input.TellWidth();
        int height = input.TellHeight();

        BMP output;
        output.SetSize(width, height);
        output.SetBitDepth(24); // RGB

        // Copy input to output to preserve borders
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                *output(x, y) = *input(x, y);

        blur_image(input, output);

        output.WriteToFile("blurred.bmp");
        return 0;
    }

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
    if (!input.ReadFromFile("image.bmp")) {
        std::cerr << "Could not open image.bmp\n";
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

.. code-block:: c++

    void blur_image(const BMP& input, BMP& output)

The function begins by querying the dimensions of the image:

.. code-block:: c++

    int width = input.TellWidth();
    int height = input.TellHeight();

These values are used to determine the iteration bounds of the blur operation.

Defining the iteration space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

|hpx| parallel algorithms operate over ranges. In this example, the blur is
applied only to the interior rows of the image, excluding the first and last
rows to avoid out-of-bounds accesses.

A vector of row indices is constructed as follows:

.. code-block:: c++

    std::vector<int> rows(height - 2);
    for (int i = 0; i < height - 2; ++i)
        rows[i] = i + 1;

Each element of the ``rows`` vector corresponds to a valid row index in the
range ``[1, height - 2]``. This guarantees that for every processed pixel, both
the pixel above and the pixel below exist.

Parallel blur using ``par_simd``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The blur computation is expressed using :cpp:func:`hpx::for_each` with the
:cpp:var:`hpx::execution::par_simd` execution policy:

.. code-block:: c++

    hpx::for_each(
        hpx::execution::par_simd,
        rows.begin(), rows.end(),
        [&](int y) {
            // process row y
        }
    );

The ``par_simd`` execution policy enables HPX to distribute iterations across
multiple worker threads while also applying SIMD vectorization within each
thread. Each iteration of the loop processes a single row of the image.

Processing pixels within a row
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inside the parallel loop, the code iterates over all columns of the current row:

.. code-block:: c++

    for (int x = 0; x < width; ++x) {

For each pixel at position ``(x, y)``, three neighboring pixels are accessed:

.. code-block:: c++

    const RGBApixel* top = input(x, y - 1);
    const RGBApixel* mid = input(x, y);
    const RGBApixel* bot = input(x, y + 1);

These correspond to the pixel above, the current pixel, and the pixel below. The
blur is computed using these three values, forming a simple vertical 3-point
stencil.

Blurring individual color channels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The blur operation is applied independently to each color channel (red, green,
and blue). A small helper lambda computes a weighted average of three input
values:

.. code-block:: c++

    auto blur_channel = [](int a, int b, int c) {
        return std::clamp(
            static_cast<int>(0.25f * a + 0.5f * b + 0.25f * c), 0, 255);
    };

This lambda implements the blur formula and clamps the result to the valid range
for image pixels. It is then used to update each color channel of the output
image:

.. code-block:: c++

    output(x, y)->Red   = blur_channel(top->Red,   mid->Red,   bot->Red);
    output(x, y)->Green = blur_channel(top->Green, mid->Green, bot->Green);
    output(x, y)->Blue  = blur_channel(top->Blue,  mid->Blue,  bot->Blue);

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
