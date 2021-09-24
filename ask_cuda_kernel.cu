#include <torch/extension.h>

#include <cuda.h>

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;

inline int GET_BLOCKS(const int N)
{
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

// ASK_7: [[1,2,4,5,6,8,9],[2,3,4,5,6,7,8]]
template <typename scalar_t>
__global__ void ask_d2_1245689_2345678_cuda_forward_kernel(const int num_kernels,
					const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
					const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights,
					torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> output,
					const int output_channels, const int input_channels, const int input_height, const int input_width, const int output_height, const int output_width, const int stride)
{
	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		const int px =  index % output_width;
		const int py = (index / output_width) % output_height;
		const int c  = (index / output_width  / output_height) % input_channels;
		const int n  = (index / output_width  / output_height  / input_channels) % output_channels;
		const int b  =  index / output_width  / output_height  / input_channels  / output_channels;
		const int opx = stride * px;
		const int opy = stride * py;
		const scalar_t data1 = (opx>0&&opy>0) ? input[b][c][opy-1][opx-1] : 0;
		const scalar_t data2 = opy>0 ? input[b][c][opy-1][opx] : 0;
		const scalar_t data3 = (opx<(input_width-1)&&opy>0) ? input[b][c][opy-1][opx+1] : 0;
		const scalar_t data4 = opx>0 ? input[b][c][opy][opx-1] : 0;
		const scalar_t data5 = input[b][c][opy][opx];
		const scalar_t data6 = opx<(input_width-1) ? input[b][c][opy][opx+1] : 0;
		const scalar_t data7 = (opx>0&&opy<(input_height-1)) ? input[b][c][opy+1][opx-1] : 0;
		const scalar_t data8 = opy<(input_height-1) ? input[b][c][opy+1][opx] : 0;
		const scalar_t data9 = (opx<(input_width-1)&&opy<(input_height-1)) ? input[b][c][opy+1][opx+1] : 0;
		output[b][n][c][py][px] = data1 * weights[n][c] + \
								  data2 * weights[n][c+input_channels] + \
								  data4 * weights[n][c+2*input_channels] + \
								  data5 * weights[n][c+3*input_channels] + \
								  data6 * weights[n][c+4*input_channels] + \
								  data8 * weights[n][c+5*input_channels] + \
								  data9 * weights[n][c+6*input_channels];
		output[b][n+output_channels][c][py][px] = data2 * weights[n][c+7*input_channels] + \
								  data3 * weights[n][c+8*input_channels] + \
								  data4 * weights[n][c+9*input_channels] + \
								  data5 * weights[n][c+10*input_channels] + \
								  data6 * weights[n][c+11*input_channels] + \
								  data7 * weights[n][c+12*input_channels] + \
								  data8 * weights[n][c+13*input_channels];
	}
}
torch::Tensor ask_d2_1245689_2345678_cuda_forward(torch::Tensor input, torch::Tensor weights, const int stride)
{
	const auto batch_size = input.size(0);
	const auto input_channels = input.size(1);
	const auto input_height = input.size(2);
	const auto input_width = input.size(3);
	const auto output_channels = weights.size(0);
	const auto output_height = (input_height - 1) / stride + 1;
	const auto output_width = (input_width - 1) / stride + 1;
	auto output = at::zeros( {batch_size, output_channels*2, input_channels, output_height, output_width}, input.options() );
	const int num_kernels = batch_size * output_channels * input_channels * output_height * output_width;
	AT_DISPATCH_FLOATING_TYPES(input.type(), "ask_d2_1245689_2345678_cuda_forward", ([&] {
		ask_d2_1245689_2345678_cuda_forward_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
			num_kernels,
			input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
			output_channels, input_channels, input_height, input_width, output_height, output_width, stride);
	}));
	return at::sum(output, 2);
}
// ASK_6: [[1,2,4,5,6,8],[2,3,4,5,6,8],[2,4,5,6,7,8],[2,4,5,6,8,9]]
template <typename scalar_t>
__global__ void ask_d4_124568_234568_245678_245689_cuda_forward_kernel(const int num_kernels,
					const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
					const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights,
					torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> output,
					const int output_channels, const int input_channels, const int input_height, const int input_width, const int output_height, const int output_width, const int stride)
{
	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		const int px =  index % output_width;
		const int py = (index / output_width) % output_height;
		const int c  = (index / output_width  / output_height) % input_channels;
		const int n  = (index / output_width  / output_height  / input_channels) % output_channels;
		const int b  =  index / output_width  / output_height  / input_channels  / output_channels;
		const int opx = stride * px;
		const int opy = stride * py;
		const scalar_t data1 = (opx>0&&opy>0) ? input[b][c][opy-1][opx-1] : 0;
		const scalar_t data2 = opy>0 ? input[b][c][opy-1][opx] : 0;
		const scalar_t data3 = (opx<(input_width-1)&&opy>0) ? input[b][c][opy-1][opx+1] : 0;
		const scalar_t data4 = opx>0 ? input[b][c][opy][opx-1] : 0;
		const scalar_t data5 = input[b][c][opy][opx];
		const scalar_t data6 = opx<(input_width-1) ? input[b][c][opy][opx+1] : 0;
		const scalar_t data7 = (opx>0&&opy<(input_height-1)) ? input[b][c][opy+1][opx-1] : 0;
		const scalar_t data8 = opy<(input_height-1) ? input[b][c][opy+1][opx] : 0;
		const scalar_t data9 = (opx<(input_width-1)&&opy<(input_height-1)) ? input[b][c][opy+1][opx+1] : 0;
		output[b][n][c][py][px] = data1 * weights[n][c] + \
								  data2 * weights[n][c+input_channels] + \
								  data4 * weights[n][c+2*input_channels] + \
								  data5 * weights[n][c+3*input_channels] + \
								  data6 * weights[n][c+4*input_channels] + \
								  data8 * weights[n][c+5*input_channels];
		output[b][n+output_channels][c][py][px] = data2 * weights[n][c+6*input_channels] + \
								  data3 * weights[n][c+7*input_channels] + \
								  data4 * weights[n][c+8*input_channels] + \
								  data5 * weights[n][c+9*input_channels] + \
								  data6 * weights[n][c+10*input_channels] + \
								  data8 * weights[n][c+11*input_channels];
		output[b][n+2*output_channels][c][py][px] = data2 * weights[n][c+12*input_channels] + \
								  data4 * weights[n][c+13*input_channels] + \
								  data5 * weights[n][c+14*input_channels] + \
								  data6 * weights[n][c+15*input_channels] + \
								  data7 * weights[n][c+16*input_channels] + \
								  data8 * weights[n][c+17*input_channels];
		output[b][n+3*output_channels][c][py][px] = data2 * weights[n][c+18*input_channels] + \
								  data4 * weights[n][c+19*input_channels] + \
								  data5 * weights[n][c+20*input_channels] + \
								  data6 * weights[n][c+21*input_channels] + \
								  data8 * weights[n][c+22*input_channels] + \
								  data9 * weights[n][c+23*input_channels];
	}
}
torch::Tensor ask_d4_124568_234568_245678_245689_cuda_forward(torch::Tensor input, torch::Tensor weights, const int stride)
{
	const auto batch_size = input.size(0);
	const auto input_channels = input.size(1);
	const auto input_height = input.size(2);
	const auto input_width = input.size(3);
	const auto output_channels = weights.size(0);
	const auto output_height = (input_height - 1) / stride + 1;
	const auto output_width = (input_width - 1) / stride + 1;
	auto output = at::zeros( {batch_size, output_channels*4, input_channels, output_height, output_width}, input.options() );
	const int num_kernels = batch_size * output_channels * input_channels * output_height * output_width;
	AT_DISPATCH_FLOATING_TYPES(input.type(), "ask_d4_124568_234568_245678_245689_cuda_forward", ([&] {
		ask_d4_124568_234568_245678_245689_cuda_forward_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
			num_kernels,
			input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
			output_channels, input_channels, input_height, input_width, output_height, output_width, stride);
	}));
	return at::sum(output, 2);
}
// ASK_5a: [[2,4,5,6,8]]
template <typename scalar_t>
__global__ void ask_d1_24568_cuda_forward_kernel(const int num_kernels,
					const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
					const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights,
					torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> output,
					const int output_channels, const int input_channels, const int input_height, const int input_width, const int output_height, const int output_width, const int stride)
{
	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		const int px =  index % output_width;
		const int py = (index / output_width) % output_height;
		const int c  = (index / output_width  / output_height) % input_channels;
		const int n  = (index / output_width  / output_height  / input_channels) % output_channels;
		const int b  =  index / output_width  / output_height  / input_channels  / output_channels;
		const int opx = stride * px;
		const int opy = stride * py;
		const scalar_t data2 = opy>0 ? input[b][c][opy-1][opx] : 0;
		const scalar_t data4 = opx>0 ? input[b][c][opy][opx-1] : 0;
		const scalar_t data5 = input[b][c][opy][opx];
		const scalar_t data6 = opx<(input_width-1) ? input[b][c][opy][opx+1] : 0;
		const scalar_t data8 = opy<(input_height-1) ? input[b][c][opy+1][opx] : 0;
		output[b][n][c][py][px] = data2 * weights[n][c] + \
								  data4 * weights[n][c+input_channels] + \
								  data5 * weights[n][c+2*input_channels] + \
								  data6 * weights[n][c+3*input_channels] + \
								  data8 * weights[n][c+4*input_channels];
	}
}
torch::Tensor ask_d1_24568_cuda_forward(torch::Tensor input, torch::Tensor weights, const int stride)
{
	const auto batch_size = input.size(0);
	const auto input_channels = input.size(1);
	const auto input_height = input.size(2);
	const auto input_width = input.size(3);
	const auto output_channels = weights.size(0);
	const auto output_height = (input_height - 1) / stride + 1;
	const auto output_width = (input_width - 1) / stride + 1;
	auto output = at::zeros( {batch_size, output_channels*1, input_channels, output_height, output_width}, input.options() );
	const int num_kernels = batch_size * output_channels * input_channels * output_height * output_width;
	AT_DISPATCH_FLOATING_TYPES(input.type(), "ask_d1_24568_cuda_forward", ([&] {
		ask_d1_24568_cuda_forward_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
			num_kernels,
			input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
			output_channels, input_channels, input_height, input_width, output_height, output_width, stride);
	}));
	return at::sum(output, 2);
}
// ASK_5b: [[2,4,5,6,8],[2,4,5,6,8],[2,4,5,6,8],[1,3,5,7,9]]
template <typename scalar_t>
__global__ void ask_d4_24568_24568_24568_13579_cuda_forward_kernel(const int num_kernels,
					const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
					const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights,
					torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> output,
					const int output_channels, const int input_channels, const int input_height, const int input_width, const int output_height, const int output_width, const int stride)
{
	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		const int px =  index % output_width;
		const int py = (index / output_width) % output_height;
		const int c  = (index / output_width  / output_height) % input_channels;
		const int n  = (index / output_width  / output_height  / input_channels) % output_channels;
		const int b  =  index / output_width  / output_height  / input_channels  / output_channels;
		const int opx = stride * px;
		const int opy = stride * py;
		const scalar_t data1 = (opx>0&&opy>0) ? input[b][c][opy-1][opx-1] : 0;
		const scalar_t data2 = opy>0 ? input[b][c][opy-1][opx] : 0;
		const scalar_t data3 = (opx<(input_width-1)&&opy>0) ? input[b][c][opy-1][opx+1] : 0;
		const scalar_t data4 = opx>0 ? input[b][c][opy][opx-1] : 0;
		const scalar_t data5 = input[b][c][opy][opx];
		const scalar_t data6 = opx<(input_width-1) ? input[b][c][opy][opx+1] : 0;
		const scalar_t data7 = (opx>0&&opy<(input_height-1)) ? input[b][c][opy+1][opx-1] : 0;
		const scalar_t data8 = opy<(input_height-1) ? input[b][c][opy+1][opx] : 0;
		const scalar_t data9 = (opx<(input_width-1)&&opy<(input_height-1)) ? input[b][c][opy+1][opx+1] : 0;
		output[b][n][c][py][px] = data2 * weights[n][c] + \
								  data4 * weights[n][c+input_channels] + \
								  data5 * weights[n][c+2*input_channels] + \
								  data6 * weights[n][c+3*input_channels] + \
								  data8 * weights[n][c+4*input_channels];
		output[b][n+output_channels][c][py][px] = data2 * weights[n][c+5*input_channels] + \
								  data4 * weights[n][c+6*input_channels] + \
								  data5 * weights[n][c+7*input_channels] + \
								  data6 * weights[n][c+8*input_channels] + \
								  data8 * weights[n][c+9*input_channels];
		output[b][n+2*output_channels][c][py][px] = data2 * weights[n][c+10*input_channels] + \
								  data4 * weights[n][c+11*input_channels] + \
								  data5 * weights[n][c+12*input_channels] + \
								  data6 * weights[n][c+13*input_channels] + \
								  data8 * weights[n][c+14*input_channels];
		output[b][n+3*output_channels][c][py][px] = data1 * weights[n][c+15*input_channels] + \
								  data3 * weights[n][c+16*input_channels] + \
								  data5 * weights[n][c+17*input_channels] + \
								  data7 * weights[n][c+18*input_channels] + \
								  data9 * weights[n][c+19*input_channels];
	}
}
torch::Tensor ask_d4_24568_24568_24568_13579_cuda_forward(torch::Tensor input, torch::Tensor weights, const int stride)
{
	const auto batch_size = input.size(0);
	const auto input_channels = input.size(1);
	const auto input_height = input.size(2);
	const auto input_width = input.size(3);
	const auto output_channels = weights.size(0);
	const auto output_height = (input_height - 1) / stride + 1;
	const auto output_width = (input_width - 1) / stride + 1;
	auto output = at::zeros( {batch_size, output_channels*4, input_channels, output_height, output_width}, input.options() );
	const int num_kernels = batch_size * output_channels * input_channels * output_height * output_width;
	AT_DISPATCH_FLOATING_TYPES(input.type(), "ask_d4_24568_24568_24568_13579_cuda_forward", ([&] {
		ask_d4_24568_24568_24568_13579_cuda_forward_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
			num_kernels,
			input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
			output_channels, input_channels, input_height, input_width, output_height, output_width, stride);
	}));
	return at::sum(output, 2);
}
// ASK_4a: [[1,2,4,5],[2,3,5,6],[4,5,7,8],[5,6,8,9]]
template <typename scalar_t>
__global__ void ask_d4_1245_2356_4578_5689_cuda_forward_kernel(const int num_kernels,
					const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
					const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights,
					torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> output,
					const int output_channels, const int input_channels, const int input_height, const int input_width, const int output_height, const int output_width, const int stride)
{
	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		const int px =  index % output_width;
		const int py = (index / output_width) % output_height;
		const int c  = (index / output_width  / output_height) % input_channels;
		const int n  = (index / output_width  / output_height  / input_channels) % output_channels;
		const int b  =  index / output_width  / output_height  / input_channels  / output_channels;
		const int opx = stride * px;
		const int opy = stride * py;
		const scalar_t data1 = (opx>0&&opy>0) ? input[b][c][opy-1][opx-1] : 0;
		const scalar_t data2 = opy>0 ? input[b][c][opy-1][opx] : 0;
		const scalar_t data3 = (opx<(input_width-1)&&opy>0) ? input[b][c][opy-1][opx+1] : 0;
		const scalar_t data4 = opx>0 ? input[b][c][opy][opx-1] : 0;
		const scalar_t data5 = input[b][c][opy][opx];
		const scalar_t data6 = opx<(input_width-1) ? input[b][c][opy][opx+1] : 0;
		const scalar_t data7 = (opx>0&&opy<(input_height-1)) ? input[b][c][opy+1][opx-1] : 0;
		const scalar_t data8 = opy<(input_height-1) ? input[b][c][opy+1][opx] : 0;
		const scalar_t data9 = (opx<(input_width-1)&&opy<(input_height-1)) ? input[b][c][opy+1][opx+1] : 0;
		output[b][n][c][py][px] = data1 * weights[n][c] + \
								  data2 * weights[n][c+input_channels] + \
								  data4 * weights[n][c+2*input_channels] + \
								  data5 * weights[n][c+3*input_channels];
		output[b][n+output_channels][c][py][px] = data2 * weights[n][c+4*input_channels] + \
								  data3 * weights[n][c+5*input_channels] + \
								  data5 * weights[n][c+6*input_channels] + \
								  data6 * weights[n][c+7*input_channels];
		output[b][n+2*output_channels][c][py][px] = data4 * weights[n][c+8*input_channels] + \
								  data5 * weights[n][c+9*input_channels] + \
								  data7 * weights[n][c+10*input_channels] + \
								  data8 * weights[n][c+11*input_channels];
		output[b][n+3*output_channels][c][py][px] = data5 * weights[n][c+12*input_channels] + \
								  data6 * weights[n][c+13*input_channels] + \
								  data8 * weights[n][c+14*input_channels] + \
								  data9 * weights[n][c+15*input_channels];
	}
}
torch::Tensor ask_d4_1245_2356_4578_5689_cuda_forward(torch::Tensor input, torch::Tensor weights, const int stride)
{
	const auto batch_size = input.size(0);
	const auto input_channels = input.size(1);
	const auto input_height = input.size(2);
	const auto input_width = input.size(3);
	const auto output_channels = weights.size(0);
	const auto output_height = (input_height - 1) / stride + 1;
	const auto output_width = (input_width - 1) / stride + 1;
	auto output = at::zeros( {batch_size, output_channels*4, input_channels, output_height, output_width}, input.options() );
	const int num_kernels = batch_size * output_channels * input_channels * output_height * output_width;
	AT_DISPATCH_FLOATING_TYPES(input.type(), "ask_d4_1245_2356_4578_5689_cuda_forward", ([&] {
		ask_d4_1245_2356_4578_5689_cuda_forward_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
			num_kernels,
			input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
			output_channels, input_channels, input_height, input_width, output_height, output_width, stride);
	}));
	return at::sum(output, 2);
}
// ASK_4b: [[2,4,5,6],[2,5,6,8],[4,5,6,8],[2,4,5,8]]
template <typename scalar_t>
__global__ void ask_d4_2456_2568_4568_2458_cuda_forward_kernel(const int num_kernels,
					const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
					const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights,
					torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> output,
					const int output_channels, const int input_channels, const int input_height, const int input_width, const int output_height, const int output_width, const int stride)
{
	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		const int px =  index % output_width;
		const int py = (index / output_width) % output_height;
		const int c  = (index / output_width  / output_height) % input_channels;
		const int n  = (index / output_width  / output_height  / input_channels) % output_channels;
		const int b  =  index / output_width  / output_height  / input_channels  / output_channels;
		const int opx = stride * px;
		const int opy = stride * py;
		const scalar_t data2 = opy>0 ? input[b][c][opy-1][opx] : 0;
		const scalar_t data4 = opx>0 ? input[b][c][opy][opx-1] : 0;
		const scalar_t data5 = input[b][c][opy][opx];
		const scalar_t data6 = opx<(input_width-1) ? input[b][c][opy][opx+1] : 0;
		const scalar_t data8 = opy<(input_height-1) ? input[b][c][opy+1][opx] : 0;
		output[b][n][c][py][px] = data2 * weights[n][c] + \
								  data4 * weights[n][c+input_channels] + \
								  data5 * weights[n][c+2*input_channels] + \
								  data6 * weights[n][c+3*input_channels];
		output[b][n+output_channels][c][py][px] = data2 * weights[n][c+4*input_channels] + \
								  data5 * weights[n][c+5*input_channels] + \
								  data6 * weights[n][c+6*input_channels] + \
								  data8 * weights[n][c+7*input_channels];
		output[b][n+2*output_channels][c][py][px] = data4 * weights[n][c+8*input_channels] + \
								  data5 * weights[n][c+9*input_channels] + \
								  data6 * weights[n][c+10*input_channels] + \
								  data8 * weights[n][c+11*input_channels];
		output[b][n+3*output_channels][c][py][px] = data2 * weights[n][c+12*input_channels] + \
								  data4 * weights[n][c+13*input_channels] + \
								  data5 * weights[n][c+14*input_channels] + \
								  data8 * weights[n][c+15*input_channels];
	}
}
torch::Tensor ask_d4_2456_2568_4568_2458_cuda_forward(torch::Tensor input, torch::Tensor weights, const int stride)
{
	const auto batch_size = input.size(0);
	const auto input_channels = input.size(1);
	const auto input_height = input.size(2);
	const auto input_width = input.size(3);
	const auto output_channels = weights.size(0);
	const auto output_height = (input_height - 1) / stride + 1;
	const auto output_width = (input_width - 1) / stride + 1;
	auto output = at::zeros( {batch_size, output_channels*4, input_channels, output_height, output_width}, input.options() );
	const int num_kernels = batch_size * output_channels * input_channels * output_height * output_width;
	AT_DISPATCH_FLOATING_TYPES(input.type(), "ask_d4_2456_2568_4568_2458_cuda_forward", ([&] {
		ask_d4_2456_2568_4568_2458_cuda_forward_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
			num_kernels,
			input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
			output_channels, input_channels, input_height, input_width, output_height, output_width, stride);
	}));
	return at::sum(output, 2);
}
// ASK_3a: [[2,4,5],[2,5,6],[4,5,8],[5,6,8]]
template <typename scalar_t>
__global__ void ask_d4_245_256_458_568_cuda_forward_kernel(const int num_kernels,
					const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
					const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights,
					torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> output,
					const int output_channels, const int input_channels, const int input_height, const int input_width, const int output_height, const int output_width, const int stride)
{
	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		const int px =  index % output_width;
		const int py = (index / output_width) % output_height;
		const int c  = (index / output_width  / output_height) % input_channels;
		const int n  = (index / output_width  / output_height  / input_channels) % output_channels;
		const int b  =  index / output_width  / output_height  / input_channels  / output_channels;
		const int opx = stride * px;
		const int opy = stride * py;
		const scalar_t data2 = opy>0 ? input[b][c][opy-1][opx] : 0;
		const scalar_t data4 = opx>0 ? input[b][c][opy][opx-1] : 0;
		const scalar_t data5 = input[b][c][opy][opx];
		const scalar_t data6 = opx<(input_width-1) ? input[b][c][opy][opx+1] : 0;
		const scalar_t data8 = opy<(input_height-1) ? input[b][c][opy+1][opx] : 0;
		output[b][n][c][py][px] = data2 * weights[n][c] + \
								  data4 * weights[n][c+input_channels] + \
								  data5 * weights[n][c+2*input_channels];
		output[b][n+output_channels][c][py][px] = data2 * weights[n][c+3*input_channels] + \
								  data5 * weights[n][c+4*input_channels] + \
								  data6 * weights[n][c+5*input_channels];
		output[b][n+2*output_channels][c][py][px] = data4 * weights[n][c+6*input_channels] + \
								  data5 * weights[n][c+7*input_channels] + \
								  data8 * weights[n][c+8*input_channels];
		output[b][n+3*output_channels][c][py][px] = data5 * weights[n][c+9*input_channels] + \
								  data6 * weights[n][c+10*input_channels] + \
								  data8 * weights[n][c+11*input_channels];
	}
}
torch::Tensor ask_d4_245_256_458_568_cuda_forward(torch::Tensor input, torch::Tensor weights, const int stride)
{
	const auto batch_size = input.size(0);
	const auto input_channels = input.size(1);
	const auto input_height = input.size(2);
	const auto input_width = input.size(3);
	const auto output_channels = weights.size(0);
	const auto output_height = (input_height - 1) / stride + 1;
	const auto output_width = (input_width - 1) / stride + 1;
	auto output = at::zeros( {batch_size, output_channels*4, input_channels, output_height, output_width}, input.options() );
	const int num_kernels = batch_size * output_channels * input_channels * output_height * output_width;
	AT_DISPATCH_FLOATING_TYPES(input.type(), "ask_d4_245_256_458_568_cuda_forward", ([&] {
		ask_d4_245_256_458_568_cuda_forward_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
			num_kernels,
			input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
			output_channels, input_channels, input_height, input_width, output_height, output_width, stride);
	}));
	return at::sum(output, 2);
}
// ASK_3b: [[2,5,8],[4,5,6]]
template <typename scalar_t>
__global__ void ask_d2_258_456_cuda_forward_kernel(const int num_kernels,
					const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
					const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights,
					torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> output,
					const int output_channels, const int input_channels, const int input_height, const int input_width, const int output_height, const int output_width, const int stride)
{
	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		const int px =  index % output_width;
		const int py = (index / output_width) % output_height;
		const int c  = (index / output_width  / output_height) % input_channels;
		const int n  = (index / output_width  / output_height  / input_channels) % output_channels;
		const int b  =  index / output_width  / output_height  / input_channels  / output_channels;
		const int opx = stride * px;
		const int opy = stride * py;
		const scalar_t data2 = opy>0 ? input[b][c][opy-1][opx] : 0;
		const scalar_t data4 = opx>0 ? input[b][c][opy][opx-1] : 0;
		const scalar_t data5 = input[b][c][opy][opx];
		const scalar_t data6 = opx<(input_width-1) ? input[b][c][opy][opx+1] : 0;
		const scalar_t data8 = opy<(input_height-1) ? input[b][c][opy+1][opx] : 0;
		output[b][n][c][py][px] = data2 * weights[n][c] + \
								  data5 * weights[n][c+input_channels] + \
								  data8 * weights[n][c+2*input_channels];
		output[b][n+output_channels][c][py][px] = data4 * weights[n][c+3*input_channels] + \
								  data5 * weights[n][c+4*input_channels] + \
								  data6 * weights[n][c+5*input_channels];
	}
}
torch::Tensor ask_d2_258_456_cuda_forward(torch::Tensor input, torch::Tensor weights, const int stride)
{
	const auto batch_size = input.size(0);
	const auto input_channels = input.size(1);
	const auto input_height = input.size(2);
	const auto input_width = input.size(3);
	const auto output_channels = weights.size(0);
	const auto output_height = (input_height - 1) / stride + 1;
	const auto output_width = (input_width - 1) / stride + 1;
	auto output = at::zeros( {batch_size, output_channels*2, input_channels, output_height, output_width}, input.options() );
	const int num_kernels = batch_size * output_channels * input_channels * output_height * output_width;
	AT_DISPATCH_FLOATING_TYPES(input.type(), "ask_d2_258_456_cuda_forward", ([&] {
		ask_d2_258_456_cuda_forward_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
			num_kernels,
			input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
			output_channels, input_channels, input_height, input_width, output_height, output_width, stride);
	}));
	return at::sum(output, 2);
}
// ASK_2: [[2,5],[4,5],[5,6],[5,8]]
template <typename scalar_t>
__global__ void ask_d4_25_45_56_58_cuda_forward_kernel(const int num_kernels,
					const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
					const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights,
					torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> output,
					const int output_channels, const int input_channels, const int input_height, const int input_width, const int output_height, const int output_width, const int stride)
{
	CUDA_KERNEL_LOOP(index, num_kernels)
	{
		const int px =  index % output_width;
		const int py = (index / output_width) % output_height;
		const int c  = (index / output_width  / output_height) % input_channels;
		const int n  = (index / output_width  / output_height  / input_channels) % output_channels;
		const int b  =  index / output_width  / output_height  / input_channels  / output_channels;
		const int opx = stride * px;
		const int opy = stride * py;
		const scalar_t data2 = opy>0 ? input[b][c][opy-1][opx] : 0;
		const scalar_t data4 = opx>0 ? input[b][c][opy][opx-1] : 0;
		const scalar_t data5 = input[b][c][opy][opx];
		const scalar_t data6 = opx<(input_width-1) ? input[b][c][opy][opx+1] : 0;
		const scalar_t data8 = opy<(input_height-1) ? input[b][c][opy+1][opx] : 0;
		output[b][n][c][py][px] = data2 * weights[n][c] + \
								  data5 * weights[n][c+input_channels];
		output[b][n+output_channels][c][py][px] = data4 * weights[n][c+2*input_channels] + \
								  data5 * weights[n][c+3*input_channels];
		output[b][n+2*output_channels][c][py][px] = data5 * weights[n][c+4*input_channels] + \
								  data6 * weights[n][c+5*input_channels];
		output[b][n+3*output_channels][c][py][px] = data5 * weights[n][c+6*input_channels] + \
								  data8 * weights[n][c+7*input_channels];
	}
}
torch::Tensor ask_d4_25_45_56_58_cuda_forward(torch::Tensor input, torch::Tensor weights, const int stride)
{
	const auto batch_size = input.size(0);
	const auto input_channels = input.size(1);
	const auto input_height = input.size(2);
	const auto input_width = input.size(3);
	const auto output_channels = weights.size(0);
	const auto output_height = (input_height - 1) / stride + 1;
	const auto output_width = (input_width - 1) / stride + 1;
	auto output = at::zeros( {batch_size, output_channels*4, input_channels, output_height, output_width}, input.options() );
	const int num_kernels = batch_size * output_channels * input_channels * output_height * output_width;
	AT_DISPATCH_FLOATING_TYPES(input.type(), "ask_d4_25_45_56_58_cuda_forward", ([&] {
		ask_d4_25_45_56_58_cuda_forward_kernel<scalar_t><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
			num_kernels,
			input.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
			weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
			output_channels, input_channels, input_height, input_width, output_height, output_width, stride);
	}));
	return at::sum(output, 2);
}
