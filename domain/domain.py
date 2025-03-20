class Request:
    def __init__(self,prompt):
        self.prompt=prompt

    @staticmethod
    def from_json(prompt:dict):
        if "prompt" not in prompt:
            raise ValueError("prompt is not provided")
        else:
            return Request(prompt["prompt"])

class Response:
    def __init__(self,response):
        self.response=response

    def to_json(self):
        return {
            "response":self.response
        }