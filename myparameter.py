import torch 
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from typing import Dict, List
#引入基础llama模型，进行改变
import lit_llama.model as llama

from contextlib import contextmanager
from dataclasses import dataclass

in_features = 128
enable_lora = [True,False,True]
r = 2
lora_A = nn.Parameter(
                torch.zeros((r * sum(enable_lora), in_features)))
nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
a, b = lora_A.size()
print(a,b)
length = max(a,b)
minimum = min(a,b)
ran = random.sample(range(length),minimum)
x = torch.eye(length)
lora_A.requires_grad = False
for i in range(len(ran)):
    lora_A[i,:] = x[ran[i]]
print(lora_A)
print(lora_A.size())
print(sum(sum(lora_A)))