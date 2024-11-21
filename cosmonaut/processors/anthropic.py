from cosmonaut.processors.openai import OpenAIProcessor


class AnthropicProcessor(OpenAIProcessor):
    def build_messages(self, prompt, instructions):
        return [super().build_messages(prompt, instructions)[1]]

    def build_json(self, messages, temperature, instructions):
        data = super().build_json(messages, temperature, instructions)
        data["system"] = instructions
        return data

    def extract_text(self, response):
        return response["content"][0]["text"]
