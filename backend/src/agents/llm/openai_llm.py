from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from backend.utils.schema import Message
from typing import List, Dict, Any, Generator
import os
from backend.utils.logger import get_logger 
from backend.core.src.llm_tools.tools import tools_schema, function_mapping
from backend.core.src.llm.base import BaseLLM
import json
load_dotenv()

class OpenAI_LLM(BaseLLM):
    def __init__(self, model: str, logger=None):
        super().__init__(model)
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )
        self.logger = logger or get_logger()

    def chat(self, messages: List[Message]) -> str:
        """Standard chat call"""
        try:
            self.logger.info(f"Sending chat completion request to model: {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[m.to_dict() for m in messages],
            )
            content = response.choices[0].message.content
            self.logger.debug(f"Received response: {content}")
            return content
        except OpenAIError as e:
            self.logger.exception(f"Chat completion failed with Error: {e}")
            return f"Error: {e}"

    def chat_streaming(self, messages: List[Message]):
        """Streaming chat"""
        try:
            self.logger.info(f"Streaming chat request to model: {self.model}")
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[m.to_dict() for m in messages],
                stream=True,
            )
            response = ""
            for chunk in stream:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    response += delta.content
                    yield delta.content
            self.logger.debug("Streaming complete.")
        except OpenAIError as e:
            self.logger.exception(f"Streaming chat failed with Error: {e}")
            yield f"Error: {e}"
    
    def chat_with_tools(self, messages: List[Message]) -> Generator[str, None, None]:
        """
        Manages the conversation with the LLM. If a tool is needed, it's called
        and the final response is streamed. If not, the generator finishes immediately.
        """
        # Convert message objects to dictionaries for the API call
        message_dicts = [m.to_dict() for m in messages]

        try:
            # First call to the model to see if it wants to use a tool
            print("--- Calling LLM to see if a tool should be used... ---")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=message_dicts,
                tools=tools_schema,
                tool_choice="auto",
            )
            response_message = response.choices[0].message
            tool_outputs = []
            if response_message.tool_calls is not None:
                # --- Execute Tools ---
                print(f"--- LLM requested {len(response_message.tool_calls)} tool call(s) ---")
                message_dicts.append(response_message)
                
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = function_mapping.get(function_name)
                    
                    if not function_to_call:
                        output = f"Error: Tool '{function_name}' not found."
                    else:
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                            print(f"Calling function: '{function_name}' with args: {function_args}")
                            
                            if "symbols" in function_args:
                                function_args["symbols"] = set(function_args["symbols"])

                            results, failed = function_to_call(**function_args)
                            output = json.dumps({
                                "successful_results": results,
                                "failed_symbols": list(failed)
                            })

                        except Exception as e:
                            print(f"Error executing tool {function_name}: {e}")
                            output = f"Error: {e}"

                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": output,
                    })

                message_dicts.extend(tool_outputs)

            # --- Second call to get the final, streamed natural language response ---
            print("\n--- Calling LLM again with tool results... ---")
            final_response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=message_dicts,
                tools=tools_schema,
                stream=True,
            )
            
            for chunk in final_response_stream:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield delta.content
                    
        except OpenAIError as e:
            yield f"Error: {e}"
        except Exception as e:
            print(f"An error occurred in the chat process: {e}")
            yield f"An error occurred: {e}"



