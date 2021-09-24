#include <torch/extension.h>

torch::Tensor ask_d2_1245689_2345678_cuda_forward(torch::Tensor input, torch::Tensor weights, const int stride);
torch::Tensor ask_d4_124568_234568_245678_245689_cuda_forward(torch::Tensor input, torch::Tensor weights, const int stride);
torch::Tensor ask_d1_24568_cuda_forward(torch::Tensor input, torch::Tensor weights, const int stride);
torch::Tensor ask_d4_24568_24568_24568_13579_cuda_forward(torch::Tensor input, torch::Tensor weights, const int stride);
torch::Tensor ask_d4_1245_2356_4578_5689_cuda_forward(torch::Tensor input, torch::Tensor weights, const int stride);
torch::Tensor ask_d4_2456_2568_4568_2458_cuda_forward(torch::Tensor input, torch::Tensor weights, const int stride);
torch::Tensor ask_d4_245_256_458_568_cuda_forward(torch::Tensor input, torch::Tensor weights, const int stride);
torch::Tensor ask_d2_258_456_cuda_forward(torch::Tensor input, torch::Tensor weights, const int stride);
torch::Tensor ask_d4_25_45_56_58_cuda_forward(torch::Tensor input, torch::Tensor weights, const int stride);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ask_d2_1245689_2345678_cuda_forward", &ask_d2_1245689_2345678_cuda_forward, "ask forward (CUDA)");
  m.def("ask_d4_124568_234568_245678_245689_cuda_forward", &ask_d4_124568_234568_245678_245689_cuda_forward, "ask forward (CUDA)");
  m.def("ask_d1_24568_cuda_forward", &ask_d1_24568_cuda_forward, "ask forward (CUDA)");
  m.def("ask_d4_24568_24568_24568_13579_cuda_forward", &ask_d4_24568_24568_24568_13579_cuda_forward, "ask forward (CUDA)");
  m.def("ask_d4_1245_2356_4578_5689_cuda_forward", &ask_d4_1245_2356_4578_5689_cuda_forward, "ask forward (CUDA)");
  m.def("ask_d4_2456_2568_4568_2458_cuda_forward", &ask_d4_2456_2568_4568_2458_cuda_forward, "ask forward (CUDA)");
  m.def("ask_d4_245_256_458_568_cuda_forward", &ask_d4_245_256_458_568_cuda_forward, "ask forward (CUDA)");
  m.def("ask_d2_258_456_cuda_forward", &ask_d2_258_456_cuda_forward, "ask forward (CUDA)");
  m.def("ask_d4_25_45_56_58_cuda_forward", &ask_d4_25_45_56_58_cuda_forward, "ask forward (CUDA)");
}
