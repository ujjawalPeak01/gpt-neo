import json
from transformers import pipeline


class InferlessPythonModel:

    def initialize(self):
        self.generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M",device=0)

        volume_path = "/var/nfs-mount/testing-volume"
        file_path = "/var/nfs-mount/testing-volume/testing.txt"
        with open(file_path, 'w') as file:
            file.write('This is testing')
        
        print("This is Initialize Code", flush=True)
        pint("THis should fail voila")

    
    def infer(self, inputs):
        prompt = inputs["prompt"]
        pipeline_output = self.generator(prompt, do_sample=True, min_length=20, max_length=300)
        generated_txt = pipeline_output[0]["generated_text"]
        return {"generated_text": generated_txt}

    def finalize(self,args):
        self.pipe = None
