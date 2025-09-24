# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import asyncio
import inspect
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError
from typing import List, Dict, Any

from app.agent_dispatcher.domain.plan.action.skill.mcp.engine import MCPEngine
from app.agent_dispatcher.infrastructure.entity.AgentInstance import AgentInstance
from app.manus.agent.base.skill_to_tool import convert_skill_to_tool
from app.manus.llm.chat_llm import ChatLLM
from app.manus.task.time_record_util import time_record
from app.manus.task.todolist import Plan


class BaseAgent:
    def __init__(self, agent_instance: AgentInstance, llm: ChatLLM, functions: {}):
        self.agent_instance = agent_instance
        self.llm = llm
        self.tools = []
        self.mcp_tools = []
        # self.tools = [convert_skill_to_tool(skill.model_dump(), 'en') for skill in self.agent_instance.template.skills]
        for skill in self.agent_instance.template.skills:
            self.tools.extend(convert_skill_to_tool(skill.model_dump(), 'en'))
        # self.tools.extend(convert_mcp_tools(self.mcp_tools))
        self.functions = functions
        self.history = []

    def find_mcp_tool(self, tool_name):
        for tool in self.mcp_tools:
            for func in tool['mcp_tools']:
                if func.name == tool_name:
                    return tool, func.name
        return None

    def execute(self, messages: List[Dict[str, Any]], step_index=None, plan: Plan = None, max_iteration=10):
        for i in range(max_iteration):
            # print(f"messages:{messages}")
            try:
                response = self.llm.create_with_tools(messages, self.tools)

            except Exception as e:
                print(f"iter {i} execute error: {e}")
                messages[-1]["content"]=f"{e},若要读取文件，使用python代码解析和正则匹配"
                continue
            print(f"index: {i}, response:{response}")

            # Process initial response
            result = self._process_response(response, messages, step_index, plan)
            if result:
                return result

            print(f"iter {i} for {self.agent_instance.instance_name}")

        if max_iteration > 1:
            return self._handle_max_iteration(messages, step_index)
        return messages[-1].get("content")

    def _process_response(self, response, messages, step_index, plan: Plan = None, ):
        if not response.tool_calls:
            messages.append({"role": "assistant", "content": response.content})
            return response.content

        messages.append({
            "role": "assistant",
            "content": response.content,
            "tool_calls": response.tool_calls
        })

        results = self._execute_tool_calls(response.tool_calls, step_index, plan)
        messages.extend(results)

        # Check for termination conditions
        for result in results:
            if result["name"] in ["terminate", "mark_step"]:
                return result["content"]
        return None

    def _execute_tool_calls(self, tool_calls, step_index, plan: Plan = None, ):
        results = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = tool_call.function.arguments

                if function_name in self.functions:
                    futures.append(executor.submit(
                        self._execute_tool_call,
                        function_name=function_name,
                        function_args=function_args,
                        tool_call_id=tool_call.id,
                        step_index=step_index,
                        plan=plan
                    ))
                else:
                    futures.append(executor.submit(
                        self._execute_mcp_tool_call,
                        function_name=function_name,
                        function_args=function_args,
                        tool_call_id=tool_call.id,
                        plan=plan
                    ))

            for future in futures:
                try:
                    result = future.result()
                    # 创建新的结果字典，排除function_args
                    result_for_append = {
                        "role": result["role"],
                        "name": result["name"],
                        "content": result["content"],
                        "tool_call_id": result["tool_call_id"]
                    }
                    results.append(result_for_append)
                    
                    # Record tool execution in plan if available
                    try:
                        if step_index is not None and plan:
                            # 解析工具参数
                            json_function_args = result.get('function_args')
                            json_str = json_function_args if json_function_args else '{}'
                            args_dict = json.loads(json_str)
                            plan.record_tool_execution(
                                step_index=step_index,
                                tool_name=result['name'],
                                tool_args=args_dict,
                                result=result['content']
                            )
                    except JSONDecodeError as e:
                        print(f"Error recording tool execution: {e}, function_args={json_function_args}  {traceback.format_exc()}")
                    except Exception as e:
                        print(f"Error recording tool execution: {e},{traceback.format_exc()}")

                except Exception as e:
                    results.append({
                        "role": "tool",
                        "name": function_name,
                        "tool_call_id": tool_call.id,
                        "content": f"Execution error: {str(e)}"
                    })
        return results

    def _handle_max_iteration(self, messages, step_index):
        messages.append({"role": "user", "content": "Summarize the above conversation, use mark_step to mark the step"})
        mark_step_tools = [tool for tool in self.tools if tool['function']['name'] == 'mark_step']
        response = self.llm.create_with_tools(messages, mark_step_tools)
        print(f"max_iteration response:{response}")

        result = self._process_response(response, messages, step_index)
        if result:
            return result

        return messages[-1].get("content")

    @time_record
    def _execute_tool_call(self, function_name="", function_args="", tool_call_id="", step_index=None, plan: Plan = None):
        try:
            # Clean and validate JSON
            cleaned_args = function_args.replace('\\\'', '\'')
            args_dict = json.loads(cleaned_args or "{}")

            # Add step_index to args_dict if provided and not already present
            if step_index is not None and 'step_index' not in args_dict and function_name in ['mark_step']:
                args_dict['step_index'] = step_index

            function_to_call = self.functions[function_name]

            # 检查是否是异步函数
            if inspect.iscoroutinefunction(function_to_call):
                # 创建新的事件循环来运行异步函数
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(function_to_call(**args_dict))
                finally:
                    loop.close()
            else:
                # 同步函数直接调用
                result = function_to_call(**args_dict)

            return {
                "role": "tool",
                "name": function_name,
                "content": str(result),
                "tool_call_id": tool_call_id,
                "function_args": function_args
            }
        except Exception as e:
            return {
                "role": "tool",
                "name": function_name,
                "tool_call_id": tool_call_id,
                "content": f"Execution error: {str(e)}"
            }

    @time_record
    def _execute_mcp_tool_call(self, function_name="", function_args="", tool_call_id="", plan: Plan = None):
        try:
            mcp_tool, tool_name = self.find_mcp_tool(function_name)
            if mcp_tool and tool_name:
                cleaned_args = function_args.replace('\\\'', '\'')
                args_dict = json.loads(cleaned_args or "{}")
                result = asyncio.run(
                    MCPEngine.invoke_mcp_tool(mcp_tool['mcp_name'], mcp_tool['mcp_config'], tool_name,
                                              args_dict))
                return {
                    "role": "tool",
                    "name": function_name,
                    "content": str(result),
                    "tool_call_id": tool_call_id,
                    "function_args": function_args
                }
            else:
                return {
                    "role": "tool",
                    "name": function_name,
                    "tool_call_id": tool_call_id,
                    "content": f"Function {function_name} not found in available functions"
                }
        except Exception as e:
            return {
                "role": "tool",
                "name": function_name,
                "tool_call_id": tool_call_id,
                "content": f"Execution error: {str(e)}"
            }
