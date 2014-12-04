#include <iostream>
#include <algorithm>

#include <ocl_wrapper.h>
#include <utl_utils.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif


namespace kernel_strings {

const std::string kernels =
R"(
template<class Type>
__kernel void copy(int rows, int cols, __global Type *dst, __global Type *src)
{
	int col = get_global_id(0); // x
	int row = get_global_id(1); // y

	if(col >= cols || row >= rows) return;

	int index = col + row*cols;

    dst[index] = src[index];
}

template<class Type>
__kernel void addc(int rows, int cols, __global Type *dst, __global Type *src, Type c)
{
	int col = get_global_id(0); // x
	int row = get_global_id(1); // y

	if(col >= cols || row >= rows) return;

	int index = col + row*cols;

    dst[index] = src[index] + c;
}

template<class Type>

__kernel void multi(int rows, int cols, __global Type *dst, __global *src1 , __global *src2)
{
	int col = get_global_id(0); //x
	int row = get_global_id(1); //y
	int sol =0;

	if(col >= cols || row >= rows) return;
	
	int index = col + row*cols;

	for(int x= 0; x < rows; x++;){
		sol+= src1[row * cols + x]* src2[x * cols + col];
	}
	dst[index]= sol;
}
}";
}

int main()
{

    ocl::Platform platform(ocl::device_type::GPU);
    ocl::Device device = platform.device(ocl::device_type::GPU);

    // creates a context for a decice or platform
    ocl::Context context(device);

    // insert contexts into the platform
    platform.insert(context);

    // create command queue.
    ocl::Queue queue(context, device);


    // create program on a context.
    ocl::Program program(context, utl::type::Single | utl::type::Int);

    // insert kernels into the program.
    program << kernel_strings::kernels;

    // kernels are created and program is built for the context.
    program.build();

    {
        typedef float Type;
//         typedef utl::Matrix <Type,utl::column_major_tag> Matrix;
		typedef utl::Ones <Type,utl::column_major_tag> Ones;
        typedef utl::Zeros <Type,utl::column_major_tag> Zeros;
//		 typedef utl::Rand <Type,utl::column_major_tag, utl::uniform_dist_tag> Rand;

        // get the kernels.
		ocl::Kernel &kernel = program.kernel("multi", utl::type::Single);

		size_t rows = 1<<5, cols = 1<<7;

        size_t elements = rows * cols;
        size_t size_bytes = elements * sizeof(Type);

        // set the index space for the kernels
		// WorkGroupSize (x,y) = (16,16)
		// GlobalSize (x,y) = (Rows + Rows%16, Cols + Cols%16)
		kernel.setWorkSize(16, 16, cols, cols);

        // create host matrices
		auto h_matrix_in1  = Ones(rows,cols);
		auto h_matrix_in2  = Ones(cols,rows);
        auto h_matrix_out = Zeros(cols,cols);

//		std::cout << "Matrix(col_major) before calling copy kernel: " << std::endl << h_matrix_out << std::endl;

        // create device buffers on the specified context
        ocl::Buffer d_matrix_in1(context, size_bytes);
        ocl::Buffer d_matrix_in2(context, size_bytes);
        ocl::Buffer d_matrix_out(context, size_bytes);

        // copy data from host buffers to device buffers
        d_matrix_in1.write(queue, 0, h_matrix_in.data(), size_bytes);
        d_matrix_in2.write(queue, 0, h_matrix_in.data(), size_bytes);

        // execute both kernels only if the event_write is completed.
        // note that kernel executions are always asynchronous.
        kernel(queue, int(rows), int(cols), d_matrix_out.id(), d_matrix_in1.id() , d_matrix_in2.id());
        queue.finish();

        // copy data from device buffers to host buffers
        d_matrix_out.read(queue, h_matrix_out.data(), size_bytes);
		int c=0;
        for(c=0; c < cols*cols; c++){
        	if(h_matrix_out[c]/= 32){
        		std::cout<< "error" << endl;
        		std::cout<< h_matrix_out[c] << endl;
        		c = cols*cols+5;
        	}
        }
        }
		//if( h_matrix_in == h_matrix_out)
		if(c== cols*cols){
		std::cout << "Computation was correct." << std::endl;
		}
		//else
	    //std::cout << "Computation was incorrect." << std::endl;
			
//		std::cout << "Matrix(col_major) after calling copy kernel: " << std::endl << h_matrix_out << std::endl;
    }
    {

        typedef int Type;
//         typedef utl::Matrix <Type,utl::column_major_tag> Matrix;
		 typedef utl::Ones <Type,utl::column_major_tag> Ones;
        typedef utl::Zeros <Type,utl::column_major_tag> Zeros;
//        typedef utl::Rand <Type,utl::column_major_tag, utl::uniform_dist_tag> Rand;

        // get the kernels.
		ocl::Kernel &kernel = program.kernel("addc", utl::type::Int);

		size_t rows = 1<<5, cols = 1<<6;

        size_t elements = rows * cols;
        size_t size_bytes = elements * sizeof(Type);

        // set the index space for the kernels
		// WorkGroupSize (x,y) = (16,16)
		// GlobalSize (x,y) = (Rows + Rows%16, Cols + Cols%16)
		kernel.setWorkSize(16, 16, cols, rows);

        // create host matrices
		auto h_matrix_in  = Ones(rows,cols);
        auto h_matrix_out = Zeros(rows,cols);

//		std::cout << "Matrix(col_major) before calling add kernel: " << std::endl << h_matrix_out << std::endl;

        // create device buffers on the specified context
        ocl::Buffer d_matrix_in (context, size_bytes);
        ocl::Buffer d_matrix_out(context, size_bytes);

        // copy data from host buffers to device buffers
        d_matrix_in.write(queue, 0, h_matrix_in.data(), size_bytes);

        // execute both kernels only if the event_write is completed.
        // note that kernel executions are always asynchronous.
        kernel(queue, int(rows), int(cols), d_matrix_out.id(), d_matrix_in.id(), int(1));
        queue.finish();

        // copy data from device buffers to host buffers
        d_matrix_out.read(queue, h_matrix_out.data(), size_bytes);

//		std::cout << "Matrix(col_major) after calling add kernel: " << std::endl << h_matrix_out << std::endl;

		if( (h_matrix_in + 1) == h_matrix_out)
			std::cout << "Computation was correct." << std::endl;
		else
			std::cout << "Computation was incorrect." << std::endl;


    }



	{
		typedef float Type;
//         typedef utl::Matrix <Type,utl::row_major_tag> Matrix;
		typedef utl::Ones <Type,utl::row_major_tag> Ones;
		typedef utl::Zeros <Type,utl::row_major_tag> Zeros;
//		 typedef utl::Rand <Type,utl::row_major_tag, utl::uniform_dist_tag> Rand;

		// get the kernels.
		ocl::Kernel &kernel = program.kernel("copy", utl::type::Single);

		size_t rows = 1<<5, cols = 1<<7;

		size_t elements = rows * cols;
		size_t size_bytes = elements * sizeof(Type);

		// set the index space for the kernels
		// WorkGroupSize (x,y) = (16,16)
		// GlobalSize (x,y) = (Cols + Cols%16, Rows + Rows%16)
		kernel.setWorkSize(16, 16, cols, rows);

		// create host matrices
		auto h_matrix_in  = Ones(rows,cols);
		auto h_matrix_out = Zeros(rows,cols);

//		std::cout << "Matrix(row_major) before calling copy kernel: " << std::endl << h_matrix_out << std::endl;

		// create device buffers on the specified context
		ocl::Buffer d_matrix_in (context, size_bytes);
		ocl::Buffer d_matrix_out(context, size_bytes);

		// copy data from host buffers to device buffers
		d_matrix_in.write(queue, 0, h_matrix_in.data(), size_bytes);

		// execute both kernels only if the event_write is completed.
		// note that kernel executions are always asynchronous.
		kernel(queue, int(rows), int(cols), d_matrix_out.id(), d_matrix_in.id());
		queue.finish();

		// copy data from device buffers to host buffers
		d_matrix_out.read(queue, h_matrix_out.data(), size_bytes);

		if( h_matrix_in == h_matrix_out)
			std::cout << "Computation was correct." << std::endl;
		else
			std::cout << "Computation was incorrect." << std::endl;

//		std::cout << "Matrix(row_major) after calling copy kernel: " << std::endl << h_matrix_out << std::endl;
	}
	{

		typedef int Type;
//         typedef utl::Matrix <Type,utl::row_major_tag> Matrix;
		 typedef utl::Ones <Type,utl::row_major_tag> Ones;
		typedef utl::Zeros <Type,utl::row_major_tag> Zeros;
//		typedef utl::Rand <Type,utl::row_major_tag, utl::uniform_dist_tag> Rand;

		// get the kernels.
		ocl::Kernel &kernel = program.kernel("addc", utl::type::Int);

		size_t rows = 1<<8, cols = 1<<6;

		size_t elements = rows * cols;
		size_t size_bytes = elements * sizeof(Type);

		// set the index space for the kernels
		// WorkGroupSize (x,y) = (16,16)
		kernel.setWorkSize(16, 16, cols, rows);

		// create host matrices
		auto h_matrix_in  = Ones(rows,cols);
		auto h_matrix_out = Zeros(rows,cols);

//		std::cout << "Matrix(row_major) before calling add kernel: " << std::endl << h_matrix_out << std::endl;

		// create device buffers on the specified context
		ocl::Buffer d_matrix_in (context, size_bytes);
		ocl::Buffer d_matrix_out(context, size_bytes);

		// copy data from host buffers to device buffers
		d_matrix_in.write(queue, 0, h_matrix_in.data(), size_bytes);

		// execute both kernels only if the event_write is completed.
		// note that kernel executions are always asynchronous.
		kernel(queue, int(rows), int(cols), d_matrix_out.id(), d_matrix_in.id(), int(1));
		queue.finish();

		// copy data from device buffers to host buffers
		d_matrix_out.read(queue, h_matrix_out.data(), size_bytes);

//		std::cout << "Matrix(row_major) after calling add kernel: " << std::endl << h_matrix_out << std::endl;

		if( (h_matrix_in + 1) == h_matrix_out)
			std::cout << "Computation was correct." << std::endl;
		else
			std::cout << "Computation was incorrect." << std::endl;

	}

	return 0;
}
