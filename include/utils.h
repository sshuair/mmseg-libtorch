#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <string>

//TODO: REFACTOR TO PIPLINE

// Convert the Opencv Mat to Torch Tensor
// from cv::Mat {height, width, channels} to torch::Tensor {1, channels, height, width}
torch::Tensor Mat2Tensor(const cv::Mat &img, bool unsqueeze = true, int64_t unsqueeze_dim = 0);

//Convert the torch tensor to opencv Mat
cv::Mat Tensor2Mat(const torch::Tensor& tensor);

//read the image from file path
cv::Mat ReadImg(const std::string& filepath);

//save the image to disk
void SaveImg(const cv::Mat &img, const std::string &outfp);

//inference the model
torch::Tensor inference(torch::jit::script::Module& model, const torch::Tensor& x, const std::string& device="cpu", float_t threshold=0.5);

//