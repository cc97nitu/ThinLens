#include <torch/extension.h>
#include <iostream>
#include <vector>

std::vector<at::Tensor> drift_forward(
		torch::Tensor x,
		torch::Tensor px,
		torch::Tensor y,
		torch::Tensor py,
		torch::Tensor sigma,
		torch::Tensor delta,
		torch::Tensor vR,
		torch::Tensor length) {

		// update phase space coordinates
		auto pz = torch::sqrt(torch::pow(1 + delta,2) - torch::pow(px,2) - torch::pow(py,2));

		auto newX = x + length * px / pz;
		auto newY = y + length * py / pz;
		auto newSigma = sigma + (1 - vR * (1 + delta) / pz) * length;

		return {newX, px, newY, py, newSigma, delta, vR, pz};
}

std::vector<at::Tensor> drift_backward(
		torch::Tensor gradX,
		torch::Tensor gradPx,
		torch::Tensor gradY,
		torch::Tensor gradPy,
		torch::Tensor gradSigma,
		torch::Tensor gradDelta,
		torch::Tensor gradVR,
		torch::Tensor length,
		torch::Tensor px,
		torch::Tensor py,
		torch::Tensor delta,
		torch::Tensor pz,
		torch::Tensor vR) {

		// calculate gradients
		auto lOpZ2 = length / torch::pow(pz, 2);

		auto newGradPx = gradPx + lOpZ2 * ((pz + torch::pow(px,2) / pz) * gradX + py * px / pz * gradY - vR * (1 + delta) * px / pz * gradSigma);

        	auto newGradPy = gradPy + lOpZ2 * (py * px / pz * gradX + (pz + torch::pow(py,2) / pz) * gradY - vR * (1 + delta) * py / pz * gradSigma);

		auto dPzdDelta = (1 + delta) / pz;  // dpz / dDelta
        	auto newGradDelta = gradDelta - lOpZ2 * (dPzdDelta * (px * gradX + py * gradY) - vR * (pz - (1 + delta) * dPzdDelta) * gradSigma);

        	auto newGradVR = gradVR - (1 + delta) / pz * length * gradSigma;
		auto gradLength = 1 / pz * (px * gradX + py * gradY) + (1 - vR * (1 + delta) / pz) * gradSigma;

		return {gradX, newGradPx, gradY, newGradPy, gradSigma, newGradDelta, newGradVR, gradLength};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("drift_forward", &drift_forward, "drift forward");
  m.def("drift_backward", &drift_backward, "drift backward");
}
