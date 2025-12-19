# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Put the following code in a module file (toy.py)

from typing import Dict, List, Optional
import torch
import torch.nn as nn

# STEP 1: Define the model
# The compatibility between model and the task:
# For the model to be evaluated in a generation task ("generation"), we must implement get_watermark()
# For the model to be evaluated in a detection task ("detection"), we must implement detect_watermark()
# For the model to be evaluated in an end-to-end task ("default"), we must implement both functions
class ToyWatermark:
    
    model: nn.Module
    
    def __init__(self, model: torch.nn.Module, alpha: float = 0.0001, M: float = 1.0, nbits: int = 16):
        self.model = model
        
        # A wrapper can have any additional arguments. These arguments should be set in the builder func
        self.alpha = alpha
        self.M = M
        
        # Each model should have an attribute 'nbits'. If the model does not have this attribute,
        # we must set the value `message_size` in the task. If Omniseal could not find information 
        # from either model or the task, it will raise the ValueError
        self.nbits = nbits

    @torch.inference_mode()
    def generate_watermark(
        self,
        contents: torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:
        # A generate_watermark() must specific signature:
        # Args:
        #  - contents: a torch.Tensor (with batch dimension at dim=0) or a list of torch.Tensor (each without batch dimension)
        # - message: a torch.Tensor or a numpy array
        # Return:
        # - Should have the same data type as 'contents' with the same dimension

        # A dummy implementation of watermark for demo. Here we just, we just use a constant value M
        hidden = torch.full_like(contents, self.M, dtype=contents.dtype, device=contents.device)
        return contents + self.alpha * hidden
    
    @torch.inference_mode()
    def detect_watermark(
        self,
        contents: torch.Tensor,
        detection_threshold: float = 0.0,
        message_threshold: float = 0.0,
        detection_bits: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        # A detect_watermark() must have a specific signature:
        # Args:
        #  - contents: a torch.Tensor (with batch dimension at dim=0) or a list of torch.Tensor (each without batch dimension)
        #  - message_threshold: threshold used to convert the watermark output (probability
        #    of each bits being 0 or 1) into the binary n-bit message.
        #  - detection_threshold: threshold to convert the softmax output to binary indicating
        #    the probability of the content being watermarked
        #  - detection_bits: number of bits reserved for calculating detection accuracy.
        # Returns:
        #  - a dictionary of with some keys: 
        #    - "prediction": The prediction probability of the content being watermarked or not. The dimension should be `B x 1` for batch size of `B`.
        #    - "message": The secret message of dimension `B x nbits`
        #    - "detection_bits": The list of bits reserved to calculating the detection accuracy.
        #   
        #    One of "prediction" and "detection_bits" must be provided. "message" is optional
        #    If "message" is returned, Omniseal Bench will compute message accuracy scores: "bit_acc", "word_acc", "p_value", "capacity", and "log10_p_value"
        #    Otherwise, these metrics will be skipped
        
        B = len(contents)

        # Dummy implementation
        if self.alpha == 0:
            return {
                "prediction": torch.zeros((B,), device=contents.device),  # No watermark detected
                "message": torch.zeros((B, self.nbits), device=contents.device)  # Dummy message for demonstration
            }
        return {
            "prediction": torch.ones((B,), device=contents.device),  # Watermark detected
            "message": torch.ones((B, self.nbits), dtype=contents.dtype, device=contents.device)
        }
    
    
    
# STEP 2: Define the builder function
# The function can have any parameters, but should contain at least one parameter "device" which defines which device the model object will be placed too.
# It is advisable to have the parameters() of the model class __init__() match the arguments of this function.
def build_model(alpha: float = 0.0001, M: float = 1.0, nbits: int = 16, device: str = "cpu") -> ToyWatermark:
    
    model = torch.nn.Linear(2, 24)  # no actual model, just a placeholder
    model.eval().to(device)
    return ToyWatermark(model=model, alpha=alpha, M=M, nbits=nbits)
