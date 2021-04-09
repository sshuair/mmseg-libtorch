#include "utils.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <string>

int main(int argc, char const *argv[])
{
  std::string model_fp = argv[1];
  std::string img_fp = argv[2];
  std::string out_fp = argv[3];
  auto img = ReadImg(img_fp); //read the image
  auto input_tensor = Mat2Tensor(img, true);
  input_tensor[0][0] = input_tensor[0][0].sub_(123.675).div_(58.395);
  input_tensor[0][1] = input_tensor[0][1].sub_(116.28).div_(57.12);
  input_tensor[0][2] = input_tensor[0][2].sub_(103.53).div_(57.375);

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(model_fp);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  std::cout<<input_tensor.sizes()<<std::endl;
  torch::Tensor out_tensor = inference(module, input_tensor, "cpu");
  std::cout<<out_tensor.sizes()<<std::endl;
  //  post processing
  out_tensor.squeeze_();
  auto pred = out_tensor.argmax(0).toType(torch::kByte);
  std::cout<<pred.dtype()<<std::endl;
  auto result = Tensor2Mat(pred);
  cv::imwrite(out_fp, result);
  return 0;
}
