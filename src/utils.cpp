//
// Created by sshuair on 2021/4/8.
//

#include "utils.h"
#include <string>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <torch/torch.h>
#include <torch/script.h>


cv::Mat ReadImg(const std::string& filepath){
  cv::Mat image = cv::imread(filepath);  // CV_8UC3
  if (image.empty() || !image.data) {
    throw std::invalid_argument( "read image error! check the image." );
  }
  return image;
}

torch::Tensor Mat2Tensor(const cv::Mat& img,  bool unsqueeze, int64_t unsqueeze_dim){
  torch::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols, 3 }, at::kByte);
  tensor_image = tensor_image.permute(torch::IntArrayRef  {2,0,1});
  if (unsqueeze)
  {
    tensor_image.unsqueeze_(unsqueeze_dim);
  }

  return tensor_image.toType(torch::kFloat);
}

//TODO const torch::jit::script::Module& model 为啥不行呢
//torch::Tensor inference(torch::jit::script::Module* model, const torch::Tensor& input_tensor,  const std::string& device, float_t threshold){
torch::Tensor inference(torch::jit::script::Module& model, const torch::Tensor& input_tensor,  const std::string& device, float_t threshold){
  if (device == "cpu"){
    input_tensor.to(torch::kCPU);
  }
  else if (device == "gpu"){
    input_tensor.to(torch::kCUDA);
  }
  else{
  }

  torch::Tensor out_tensor = model.forward({input_tensor}).toTensor();
  return out_tensor;
}


cv::Mat Tensor2Mat(const torch::Tensor& tensor)
{
    int width = tensor.sizes()[0];
    int height = tensor.sizes()[1];
    try
    {
        cv::Mat output_mat(cv::Size{ height, width }, CV_8UC1, tensor.data_ptr<uchar>());
        return output_mat.clone();
    }
    catch (const torch::Error& e)
    {
        std::cout << "an error has occured : " << e.msg() << std::endl;
    }
}