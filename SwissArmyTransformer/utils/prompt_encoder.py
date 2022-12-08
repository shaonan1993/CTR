import torch
from model.VQVAE import VQVAE


class PromptEncoder(torch.nn.Module):
    def __init__(self, prompt_len, hidden_size):
        super().__init__()
        self.vqvae = VQVAE(prompt_len, hidden_size)
        
        # TODO Possible Resources: 《Benchmarking Generation via In-Context Instructions on 1,600+ Language Tasks》
        self.task_des = {
            "multiQA": "",
            "sentiment": ""
        }

        self.task_list = self.task_des.keys()

        print("init VQVAE prompt encoder...")

    def forward(self, task_id=None, task_des=None):
        #TODO task_id is for training and task_des is for zero-shot.
        # TODO need to code for dataset process to add task_id and task_des

        # Zero-shot
        if task_des is not None:
            prompt = self.vqvae(task_des)
        elif task_id is not None:
            # prompt = self.vqvae(self.task_des[self.task_list[task_id]])
            raise Exception("task id method has not finished yet.") 
        else:
            raise Exception("Either task description or task id should not be None.") 

        return prompt
