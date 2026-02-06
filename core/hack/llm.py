# import asyncio
# import os
# from typing import List, Dict, Any
# from openai import AsyncOpenAI, OpenAI

# _openai_async_client = AsyncOpenAI(api_key="sk-chakshu-12345", base_url="http://159.65.152.60:4000")

# async def get_chat_completion_async(model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:

#     response = await _openai_async_client.chat.completions.create(
#         model=model,
#         messages=messages
#     )
#     return response.model_dump()


# async def main():
#     model_name = "gemini-2.5-flash-preview-04-17" 

#     async def get_completion_with_id_wrapper(req_id: str, model_name_param: str, messages_list: List[Dict[str, str]]):
#         """Helper coroutine to run a task and return its result along with its ID."""
#         try:
#             result = await get_chat_completion_async(model_name_param, messages_list)
#             return req_id, result, None  # (identifier, result_data, error_object)
#         except Exception as e:
#             return req_id, None, e # (identifier, no_result_data, error_object)

#     # Create a list of asyncio.Task objects
#     asyncio_task_list = []
#     for i in range(5):
#         request_id = f"Request {i+1}"
#         current_messages_for_task = [
#             {
#                 "role": "system",
#                 "content": "You are a helpful math tutor."
#             },
#             {
#                 "role": "user",
#                 "content": f"What is the answer to 8x + 7 = -{i}. Also give the answer in decimal form."
#             }
#         ]
#         # Create the coroutine from the wrapper
#         coro = get_completion_with_id_wrapper(request_id, model_name, current_messages_for_task)
#         # Create an asyncio.Task from the coroutine
#         task = asyncio.create_task(coro)
#         asyncio_task_list.append(task)
    
#     # Iterate through tasks as they complete
#     for i, completed_task_future in enumerate(asyncio.as_completed(asyncio_task_list)):
#         # completed_task_future is one of the Task objects from asyncio_task_list
#         try:
#             # Await the completed task to get its result, which is the (id, result, error) tuple
#             original_request_id, response_data, error_obj = await completed_task_future
            
#             if error_obj:
#                 print(f"\nError processing {original_request_id} (completed {i+1}): {error_obj}")
#             else:
#                 print(f"\nResponse for {original_request_id} (completed {i+1}):")
#                 print(response_data)
#                 if response_data and 'choices' in response_data and response_data['choices']:
#                     print(f"Content from {original_request_id}:")
#                     print(response_data['choices'][0]['message']['content'])
#                 else:
#                     print(f"No content or unexpected format for {original_request_id}")
#         except Exception as e_outer:
#             # This catches errors from awaiting completed_task_future itself,
#             # or if the unpacking fails for an unexpected reason.
#             print(f"\nOuter error during as_completed iteration (item {i+1}): {e_outer}")


# if __name__ == "__main__":
#     asyncio.run(main())


# Sequence = lambda *args: args
# Task = lambda func: func
# Parallel = lambda *args: args
# fetch = lambda: print("Fetching data...")
# parse_page = lambda: print("Parsing page...")
# fetch_metadata = lambda: print("Fetching metadata...")
# finalize = lambda: print("Finalizing...")
# store = lambda: print("Storing data...")

# pipeline = Sequence(
#     Task(fetch),
#     Parallel(
#       Task(parse_page),
#       Sequence(Task(fetch_metadata), Task(store))
#     ),
#     Task(finalize)
# )