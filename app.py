from transformers import pipeline
import time


class InferlessPythonModel:

    def initialize(self):
        self.generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M",device=0)
        print("This is Initialize Code", flush=True)

    
    def infer(self, inputs):
        start_time = time.time()
        prompt = inputs["prompt"]
        pipeline_output = self.generator(prompt, do_sample=True, min_length=20, max_length=300)
        generated_txt = pipeline_output[0]["generated_text"]
        total_time = time.time() - start_time
        print("Start Time:", start_time, flush=True)
        print("Total Infer Time:", total_time, flush=True)
        return {"generated_text": generated_txt}

    def finalize(self,args):
        self.pipe = None
