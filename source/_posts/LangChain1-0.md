---
title: LangChain 1.0 学习笔记
date: 2026-08-03 00:00:00
updated: 2026-08-03 00:00:00
categories:
  - AI
tags:
  - LangChain
  - AI
  - Agent
description: 记录 LangChain 1.x 的核心 API 与概念，涵盖 Agents、Models、Tools、Memory、Middleware、Guardrails、Runtime 与 MCP。
---

> 本文记录 LangChain 1.x 的 API 与概念。框架迭代较快，运行示例前应对照当前官方文档核对依赖版本。

# LangChain

构建基础工具、构建基础智能体

# LangGraph

提供图结构支持、构建多智能体

# Deep Agents

构建复杂智能体

# LangChain integrations packages

集成访问各种LLM的provider

提供实现各种功能的component

# LongSmith

部署、监控、测试、日志

# 快速上手

1. 安装langchain

   ```shell
   uv add langchain
   ```

   官方给的示例采用`claude-sonnet-4-5`作为基础模型，在国内使用颇为不便。

   下面的示例，我改成`DeepSeek-V3.2`，为此，需要额外安装两个依赖：

   ```
   uv add dotenv langchain-openai
   ```

   dotenv 用来[动态加载](https://so.csdn.net/so/search?q=动态加载&spm=1001.2101.3001.7020)环境变量，首先需要创建一个`.env`文件，然后将硅基流动平台[2]的 API-KEY 填进去：

   ```env
   SILICONFLOW_API_KEY=
   ```

2. 配置完后，就可以运行下面的示例，创建一个调用工具的 agent

   ```python
   import os
   from dotenv import load_dotenv
   from langchain.agents import create_agent
   from langchain_openai import ChatOpenAI

   load_dotenv()

   llm = ChatOpenAI(
       base_url="https://api.siliconflow.cn/v1",
       api_key=os.getenv("SILICONFLOW_API_KEY"),
       model="deepseek-ai/DeepSeek-V3.2-Exp",
   )


   def get_weather(city: str) -> str:
       """Get weather for a given city."""
       return f"It's always sunny in {city}!"


   agent = create_agent(
       model=llm,
       tools=[get_weather],
       system_prompt="You are a helpful assistant",
   )

   response=agent.invoke({
       "messages": [{"role": "user", "content": "what is the weather in sf"}]
   })

   print(response)
   ```

解释:

| 顺序 | 消息类型                  | 含义                             |
| ---- | ------------------------- | -------------------------------- |
| ①    | HumanMessage              | 用户提问                         |
| ②    | AIMessage（带 tool_call） | 模型决定调用工具                 |
| ③    | ToolMessage               | 工具执行的结果                   |
| ④    | AIMessage                 | 模型根据工具返回结果生成最终回答 |

# 核心概念

## Agents

智能体将大语言模型与[工具](https://docs.langchain.com/oss/python/langchain/tools)相结合，创建能够对任务进行推理、决定使用哪些工具并迭代地寻求解决方案的系统。

[`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent)提供可用于生产环境的代理实现。[LLM 智能体循环运行各种工具以实现目标](https://simonwillison.net/2025/Sep/18/agents/)。智能体持续运行，直到满足停止条件为止——即模型发出最终输出或达到迭代次数限制。

![image-20251209145659582](https://ray-blog.oss-cn-hangzhou.aliyuncs.com/images/image-20251209145659582.png)

> [`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent)使用[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)构建基于**图的**代理运行时。图由节点（步骤）和边（连接）组成，定义了代理如何处理信息。代理在这个图中移动，执行诸如模型节点（调用模型）、工具节点（执行工具）或中间件之类的节点。

model：模型是智能体的推理引擎。它可以通过多种方式进行指定，支持静态和动态模型选择。

### Model

#### 静态模型

- 静态模型在创建代理时配置一次，并在整个执行过程中保持不变。这是最常见、最直接的方法。

  ```py
  #静态模型在创建代理时配置一次，并在整个执行过程中保持不变。这是最常见、最直接的方法。
  from langchain.agents import create_agent

  agent = create_agent(
      "gpt-5",
      tools=tools
  )
  ```

  > 模型标识符字符串支持自动推理（例如，`"gpt-5"`将被推理为`"openai:gpt-5"`）。请参阅参考[文档](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model(model))以获取完整的模型标识符字符串映射列表。

  为了更好地控制模型配置，可以直接使用提供程序包初始化模型实例。在本例中，我们使用 `<path>` [`ChatOpenAI`](https://reference.langchain.com/python/integrations/langchain_openai/ChatOpenAI)。有关其他可用的聊天[模型类，请参阅“聊天模型”部分。](https://docs.langchain.com/oss/python/integrations/chat)

  ```py
  from langchain.agents import create_agent
  from langchain_openai import ChatOpenAI

  model = ChatOpenAI(
      model="gpt-5",
      temperature=0.1,
      max_tokens=1000,
      timeout=30
      # ... (other params)
  )
  agent = create_agent(model, tools=tools)
  ```

#### 动态模型

- 动态模型的选择是在运行时环境基于当前状态以及上下文信息。这使得复杂的路由逻辑和成本优化成为可能

  要使用动态模型，请使用[`@wrap_model_call`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.wrap_model_call)装饰器创建中间件，该装饰器会在请求中修改模型：

  ```py
  from langchain_openai import ChatOpenAI
  from langchain.agents import create_agent
  from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
  from dotenv import load_dotenv
  import os

  load_dotenv()
  basic_model = ChatOpenAI(model="deepseek-ai/DeepSeek-V3.2-Exp",
                           base_url="https://api.siliconflow.cn/v1",
                           api_key=os.getenv("SILICONFLOW_API_KEY")
                           )
  advanced_model = ChatOpenAI(model="deepseek-ai/DeepSeek-R1",
                              base_url="https://api.siliconflow.cn/v1",
                              api_key=os.getenv("SILICONFLOW_API_KEY")
                              )


  def get_weather(city: str) -> str:
      """Get weather for a given city."""
      return f"It's always sunny in {city}!"


  @wrap_model_call
  def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
      """Choose model based on conversation complexity."""
      message_count = len(request.state["messages"])

      if message_count > 1:
          # Use an advanced model for longer conversations
          model = advanced_model
      else:
          model = basic_model

      request.override(model=model)
      return handler(request)


  agent = create_agent(
      model=basic_model,  # Default model
      tools=[get_weather],
      middleware=[dynamic_model_selection]
  )

  response = agent.invoke({
      "messages": [
                   {"role": "user", "content": "上海最近的天气怎么样"},
                   ]
  })

  print(response)

  ```

### Tool

工具赋予智能体执行操作的能力。智能体超越了简单的仅模型工具绑定，还能实现以下功能：

- 连续调用多个工具（由单个提示触发）
- 在适当的时候并行调用工具
- 基于先前结果的动态工具选择
- 工具重试逻辑和错误处理
- 跨工具调用保持状态持久性

#### 定义工具

```py
from langchain.tools import tool
from langchain.agents import create_agent


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(model, tools=[search, get_weather])
```

#### 工具错误处理

要自定义工具错误的处理方式，请使用[`@wrap_tool_call`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.wrap_tool_call)装饰器创建中间件

```py
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"


@wrap_tool_call
def handle_tool_errors(request,handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

llm = ChatOpenAI(
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    model="deepseek-ai/DeepSeek-V3.2-Exp",
)

agent = create_agent(
    model=llm,
    tools=[search, get_weather],
    middleware=[handle_tool_errors]
)

```

### System Prompt

可以通过提供提示来控制代理处理任务的方式。该[`system_prompt`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent(system_prompt))参数可以以字符串形式提供：

```py
agent = create_agent(
    model,
    tools,
    system_prompt="You are a helpful assistant. Be concise and accurate."
)
```

如果没有[`system_prompt`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent(system_prompt))提供任务，代理将直接从消息中推断其任务。该[`system_prompt`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent(system_prompt))参数接受 a`str`或 a [`SystemMessage`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.SystemMessage)。使用 a`SystemMessage`可以让你更好地控制提示结构

```py
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage

literary_agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": "You are an AI assistant tasked with analyzing literary works.",
            },
            {
                "type": "text",
                "text": "<the entire contents of 'Pride and Prejudice'>",
                "cache_control": {"type": "ephemeral"}
            }
        ]
    )
)

result = literary_agent.invoke(
    {"messages": [HumanMessage("Analyze the major themes in 'Pride and Prejudice'.")]}
)
```

该`cache_control`字段`{"type": "ephemeral"}`告诉 Anthropic 缓存该内容块，从而降低使用同一系统提示符的重复请求的延迟和成本。

#### Dynamic system prompt

对于需要根据运行时上下文或代理状态修改系统提示的更高级用例，可以使用[中间件](https://docs.langchain.com/oss/python/langchain/middleware)。装饰[`@dynamic_prompt`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.dynamic_prompt)器会创建中间件，该中间件会根据模型请求生成系统提示

```py
from typing import TypedDict

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import dynamic_prompt, ModelRequest
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    base_url="https://api.siliconflow.cn/v1",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    model="deepseek-ai/DeepSeek-V3.2-Exp",
)


class Context(TypedDict):
    user_role: str


@tool
def web_search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt


agent = create_agent(
    model=llm,
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
)

# The system prompt will be set dynamically based on context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]},
    context={"user_role": "expert"}
)
print(result)

```

### Invocation

通过`stream`或`invoke`调用来更新state

### Structured output

在某些情况下，您可能希望代理以特定格式返回输出。LangChain 通过参数提供了结构化输出的策略`response_format`。

#### ToolStrategy

`ToolStrategy`使用人工工具调用来生成结构化输出。这适用于任何支持工具调用的模型：

```py
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
```

#### ProviderStrategy

`ProviderStrategy` uses the model provider’s native structured output generation. This is more reliable but only works with providers that support native structured output (e.g., OpenAI):

```
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="gpt-4o",
    response_format=ProviderStrategy(ContactInfo)
)
```

### Memory

会通过消息状态自动维护对话历史记录。您还可以配置客服人员使用自定义状态方案，以便在对话过程中记住其他信息。

存储在状态中的信息可以被视为智能体的`短期记忆`：

自定义状态模式必须扩展[`AgentState`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.AgentState)为`TypedDict`。

定义自定义状态的两种方式：

1. 通过中间件
2. 通过`state_schema`On[`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent)

## Models

LLM是功能强大的AI工具，能够像人类一样理解和生成文本。它们用途广泛，无需针对每项任务进行专门训练，即可撰写内容、翻译语言、撰写摘要和回答问题。除了文本生成之外，许多模型还支持：

-  [工具调用](https://docs.langchain.com/oss/python/langchain/models#tool-calling)- 调用外部工具（如数据库查询或 API 调用）并在其响应中使用结果。
-  [结构化输出](https://docs.langchain.com/oss/python/langchain/models#structured-output)——模型的响应被限制在定义的格式内。
-  [多模态](https://docs.langchain.com/oss/python/langchain/models#multimodal)——处理并返回除文本以外的数据，例如图像、音频和视频。
-  [推理](https://docs.langchain.com/oss/python/langchain/models#reasoning)——模型执行多步骤推理以得出结论。

[模型是智能体](https://docs.langchain.com/oss/python/langchain/agents)的推理引擎。它们驱动智能体的决策过程，决定调用哪些工具、如何解释结果以及何时给出最终答案。您选择的模型的质量和功能直接影响智能体的基本可靠性和性能。不同的模型擅长不同的任务——有些模型更擅长执行复杂的指令，有些模型更擅长结构化推理，还有一些模型支持更大的上下文窗口以处理更多信息。LangChain 的标准模型接口可让您访问许多不同的提供商集成，从而可以轻松地尝试和切换模型，以找到最适合您的用例的模型。

### Basic usage

模型可以通过两种方式加以利用：

1. 在使用使用agent的时候动态指定
2. 独立执行（直接调用模型来执行文本生成、分类、提取等任务）

### Initialize a model

在 LangChain 中使用独立模型的最简单方法是，使用您选择的[聊天模型提供商](https://docs.langchain.com/oss/python/integrations/chat)[`init_chat_model`](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model)初始化一个模型（示例如下）：

**openai**

```
pip install -U "langchain[openai]"
```

```py
import os
from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"] = "sk-..."

model = init_chat_model("gpt-4.1")

response = model.invoke("Why do parrots talk?")
```

### Parameters

聊天模型接受可用于配置其行为的参数。支持的参数集因模型和提供商而异，但标准参数包括：

model

string

required

聊天模型接受可用于配置其行为的参数。支持的参数集因模型和提供商而异，但标准参数包括：

api_key

string

用于向模型提供商进行身份验证的密钥。通常在您注册访问模型时颁发。通常通过设置来访问。环境变量。

temperature

number

控制模型输出的随机性。数值越高，响应越具创造性；数值越低，响应越具确定性。

max_tokens

number

限制总数代币在响应中，有效地控制输出的长度。

timeout

number

等待模型响应的最长时间（以秒为单位），超过此时间将取消请求。

max_retries

number

如果由于网络超时或速率限制等问题导致请求失败，系统将尝试重新发送请求的最大次数。

### Key methods

#### Invoke

必须调用聊天模型才能生成输出。有三种主要的调用方法，每种方法都适用于不同的使用场景。

```py
response = model.invoke("Why do parrots have colorful feathers?")
print(response)
```

#### Stream

大多数模型都能在生成输出内容的同时进行流式传输。通过逐步显示输出，流式传输显著提升了用户体验，尤其是在处理较长的响应时。调用[`stream()`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.stream)返回一个迭代器它会在生成过程中实时输出数据块。您可以使用循环来实时处理每个数据块：

```py
for chunk in model.stream("Why do parrots have colorful feathers?"):
    print(chunk.text, end="|", flush=True)
```

#### Batch

将一系列独立的模型请求批量处理，可以显著提高性能并降低成本，因为可以并行处理这些请求：

```py
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])
for response in responses:
    print(response)
```

### Tool calling

模型可以请求调用工具来执行诸如从数据库获取数据、搜索网络或运行代码等任务。工具是以下各项的组合：

1. 模式，包括工具名称、描述和/或参数定义（通常是 JSON 模式）
2. 函数或协程执行。

![image-20251226100654516](https://ray-blog.oss-cn-hangzhou.aliyuncs.com/images/image-20251226100654516.png)

工具需要被模型使用需要通过使用`bind_tools`，后续模型会自动选择合适的工具。

```py
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("What's the weather like in Boston?")
for tool_call in response.tool_calls:
    # View tool calls made by the model
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
```

绑定用户自定义工具时，模型的响应包含执行该工具的**请求。如果模型与agent分开使用，则需要您自行执行请求的工具并将结果返回给模型以供后续推理使用。如果使用agent，代理循环将自动处理工具执行循环。

- 工具执行循环

当模型返回工具调用时，您需要执行这些工具并将结果返回给模型。这会创建一个对话循环，模型可以使用工具结果生成最终响应。

- 强制工具调用

默认情况下，模型可以根据用户输入自由选择使用哪个绑定工具。但是，您可能希望强制选择某个工具，确保模型使用特定工具或给定列表中的**任意工具**

- 并行工具调用

许多模型支持在适当情况下并行调用多个工具。这使得模型能够同时从不同来源收集信息。

- 流式工具调用

在流式响应中，工具调用是逐步构建的[`ToolCallChunk`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.ToolCallChunk)。这样，您就可以在工具调用生成过程中看到它们，而无需等待完整的响应。

### Structured ouput

模型可以按照给定的模式提供响应。这有助于确保输出易于解析，并可用于后续处理。LangChain 支持多种模式类型和方法来强制输出结构化数据。

[Pydantic](https://docs.pydantic.dev/latest/concepts/models/#basic-model-usage)  提供最丰富的功能集，包括字段验证、描述和嵌套结构。

```py
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("Provide details about the movie Inception")
print(response)  # Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)
```

TypedDict 用 Python 的内置类型系统提供了一种更简单的替代方案，非常适合不需要运行时验证的情况。

```py
from typing_extensions import TypedDict, Annotated

class MovieDict(TypedDict):
    """A movie with details."""
    title: Annotated[str, ..., "The title of the movie"]
    year: Annotated[int, ..., "The year the movie was released"]
    director: Annotated[str, ..., "The director of the movie"]
    rating: Annotated[float, ..., "The movie's rating out of 10"]

model_with_structure = model.with_structured_output(MovieDict)
response = model_with_structure.invoke("Provide details about the movie Inception")
print(response)  # {'title': 'Inception', 'year': 2010, 'director': 'Christopher Nolan', 'rating': 8.8}
```

JSON模式 为了获得最大的控制权或互操作性，您可以提供原始的 JSON Schema。

```py
import json

json_schema = {
    "title": "Movie",
    "description": "A movie with details",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "The title of the movie"
        },
        "year": {
            "type": "integer",
            "description": "The year the movie was released"
        },
        "director": {
            "type": "string",
            "description": "The director of the movie"
        },
        "rating": {
            "type": "number",
            "description": "The movie's rating out of 10"
        }
    },
    "required": ["title", "year", "director", "rating"]
}

model_with_structure = model.with_structured_output(
    json_schema,
    method="json_schema",
)
response = model_with_structure.invoke("Provide details about the movie Inception")
print(response)  # {'title': 'Inception', 'year': 2010, ...}
```

## Messages

在 LangChain 中，消息是模型的基本上下文单元。它们代表模型的输入和输出，携带与 LLM 交互时表示对话状态所需的内容和元数据。消息是包含以下内容的对象：

-  [**角色**](https://docs.langchain.com/oss/python/langchain/messages#message-types)- 标识消息类型（例如`system`，`user`）
-  [**内容**](https://docs.langchain.com/oss/python/langchain/messages#message-content)——指消息的实际内容（例如文本、图像、音频、文档等）。
-  [**元数据**](https://docs.langchain.com/oss/python/langchain/messages#message-metadata)- 可选字段，例如响应信息、消息 ID 和令牌使用情况

LangChain 提供了一种适用于所有模型提供程序的标准消息类型，确保无论调用哪个模型，行为都保持一致。

### Basic usage

#### Text prompts

```py
response = model.invoke("Write a haiku about spring")
```

#### Message prompts

```py
from langchain.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage("You are a poetry expert"),
    HumanMessage("Write a haiku about spring"),
    AIMessage("Cherry blossoms bloom...")
]
response = model.invoke(messages)
```

#### Dictionary format

```py
messages = [
    {"role": "system", "content": "You are a poetry expert"},
    {"role": "user", "content": "Write a haiku about spring"},
    {"role": "assistant", "content": "Cherry blossoms bloom..."}
]
response = model.invoke(messages)
```

### Message Type

-  [系统消息](https://docs.langchain.com/oss/python/langchain/messages#system-message)——告诉模型如何运行，并为交互提供上下文。
-  [人类消息](https://docs.langchain.com/oss/python/langchain/messages#human-message)——代表用户输入以及与模型的交互
-  [AI消息](https://docs.langchain.com/oss/python/langchain/messages#ai-message)——模型生成的响应，包括文本内容、工具调用和元数据
-  [工具消息- 表示](https://docs.langchain.com/oss/python/langchain/messages#tool-message)[工具调用](https://docs.langchain.com/oss/python/langchain/models#tool-calling)的输出

## Tools

工具扩展了agent的功能——使它们能够获取实时数据、执行代码、查询外部数据库并在现实世界中采取行动。在底层，工具是具有明确定义输入和输出的可调用函数，这些函数会被传递给[聊天模型](https://docs.langchain.com/oss/python/langchain/models)。模型会根据对话上下文决定何时调用工具以及需要提供哪些输入参数。

### create tool

创建工具最简单的方法是使用[`@tool`](https://reference.langchain.com/python/langchain/tools/#langchain.tools.tool)装饰器。默认情况下，函数的文档字符串会成为工具的描述，帮助模型理解何时使用该工具：

### 自定义工具属性

#### 自定义工具名称

默认情况下，工具名称来源于函数名称。如果需要更具描述性的名称，请对其进行覆盖：

```
@tool("web_search")  # Custom name
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

print(search.name)  # web_search
```

#### 自定义工具描述

覆盖自动生成的工具描述，以便获得更清晰的模型指导：

```
@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))
```

### 访问上下文

> **重要性：**工具能够访问代理状态、运行时上下文和长期记忆时，其功能最为强大。这使得工具能够做出上下文感知决策、个性化响应，并在对话过程中保持信息畅通。运行时上下文提供了一种在运行时将依赖项（如数据库连接、用户 ID 或配置）注入到工具中的方法，使工具更易于测试和重用。

工具可以通过该参数访问运行时信息`ToolRuntime`，该参数提供：

- **状态**- 在执行过程中流动的可变数据（例如，消息、计数器、自定义字段）
- **上下文**- 不可变配置，例如用户 ID、会话详细信息或应用程序特定配置
- **存储**- 跨对话的持久长期记忆
- **流写入器**- 工具执行时流式自定义更新
- **配置**-`RunnableConfig`用于执行
- **工具调用 ID** - 当前工具调用的 ID

#### ToolRuntime

使用`ToolRuntime`此参数即可访问所有运行时信息。只需将其添加`runtime: ToolRuntime`到工具签名中，它就会自动注入，而无需暴露给 LLM。

工具可以通过以下方式访问当前图状态`ToolRuntime`：

```py
from langchain.tools import tool, ToolRuntime

# Access the current conversation state
@tool
def summarize_conversation(
    runtime: ToolRuntime
) -> str:
    """Summarize the conversation so far."""
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"

# Access custom state fields
@tool
def get_user_preference(
    pref_name: str,
    runtime: ToolRuntime  # ToolRuntime parameter is not visible to the model
) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")
```

## Short-term memory

短期记忆功能使应用程序能够记住单个线程或对话中的先前交互。

### uasge

要向agent添加短期记忆（线程级持久性），您需要`checkpointer`在创建agent时指定一个。

```py
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


agent = create_agent(
    "gpt-5",
    tools=[get_user_info],
    checkpointer=InMemorySaver(),
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    {"configurable": {"thread_id": "1"}},
)
```

### In production

在生产环境中，使用由数据库支持的检查点：

```shell
pip install langgraph-checkpoint-postgres
```

```py
from langchain.agents import create_agent

from langgraph.checkpoint.postgres import PostgresSaver


DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup() # auto create tables in PostgresSql
    agent = create_agent(
        "gpt-5",
        tools=[get_user_info],
        checkpointer=checkpointer,
    )
```

### use short-term memory

[启用短期记忆](https://docs.langchain.com/oss/python/langchain/short-term-memory#add-short-term-memory)后，长时间的对话可能会超出 LLM 的上下文窗口。常见的解决方案包括：

- 裁剪消息：移除前N条或后N条消息，在调用LLM之前。
- 删除消息：永久删除LangGraph状态中的消息。
- 消息摘要：将历史记录中的早期消息进行总结，并用摘要替换。
- 自定义策略：自己定义策略。

#### 修剪消息

大多数 LLM 都有最大支持的上下文窗口（以标记为单位）。一种决定何时截断消息的方法是统计消息历史记录中的标记数量，并在接近该限制时进行截断。如果您使用的是 LangChain，则可以使用 trim messages 工具，并指定要从列表中保留的标记数量，以及用于处理边界的条件`strategy`（例如，保留最后一个标记）。`max_tokens`要在代理中清理消息历史记录，请使用[`@before_model`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.before_model)中间件装饰器：

```py
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent = create_agent(
    your_model_here,
    tools=your_tools_here,
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

Your name is Bob. You told me that earlier.
If you'd like me to call you a nickname or use a different name, just say the word.
"""
```

#### 删除消息

您可以从图表状态中删除消息，以管理消息历史记录。当您想要删除特定消息或清除所有消息历史记录时，此功能非常有用。要从图状态中删除消息，可以使用`RemoveMessage`.要`RemoveMessage`使其正常工作，您需要在[reducer](https://docs.langchain.com/oss/python/langgraph/graph-api#reducers)中使用状态键。[`add_messages`](https://reference.langchain.com/python/langgraph/graphs/#langgraph.graph.message.add_messages)默认设置[`AgentState`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.AgentState)提供此功能。

删除特定的消息：

```py
from langchain.messages import RemoveMessage

def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
```

删除全部的消息：

```py
from langgraph.graph.message import REMOVE_ALL_MESSAGES

def delete_messages(state):
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

#### 消息摘要

如上所示，修剪或删除消息的问题在于，您可能会因消息队列的清理而丢失信息。因此，一些应用程序会受益于更复杂的方法，例如使用聊天模型来汇总消息历史记录。

![image-20251229143857046](https://ray-blog.oss-cn-hangzhou.aliyuncs.com/images/image-20251229143857046.png)

使用内置函数[`SummarizationMiddleware`](https://docs.langchain.com/oss/python/langchain/middleware#summarization)：

```py
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig


checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20)
        )
    ],
    checkpointer=checkpointer,
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

Your name is Bob!
"""
```

## Streaming

LangChain 实现了一个流式系统来显示实时更新。对于基于 LLM 构建的应用程序而言，流式传输对于提升其响应速度至关重要。通过逐步显示输出，即使在完全响应准备就绪之前也能实现，流式传输能够显著改善用户体验 (UX)，尤其是在处理 LLM 的延迟问题时。

### 概述

LangChain 的流式系统允许您将Agent运行的实时反馈呈现给您的应用程序。LangChain Streaming技术能实现哪些功能：

-  [**流式传输Agent进度**](https://docs.langchain.com/oss/python/langchain/streaming#agent-progress)——在代理执行每个步骤后获取状态更新。
-  [**流式 LLM 标记**](https://docs.langchain.com/oss/python/langchain/streaming#llm-tokens)——在生成时流式传输语言模型标记。
-  [**流式自定义更新**](https://docs.langchain.com/oss/python/langchain/streaming#custom-updates)— 发出用户定义的信号（例如，`"Fetched 10/100 records"`）。
-  [**流式传输多种模式**](https://docs.langchain.com/oss/python/langchain/streaming#stream-multiple-modes)——可选择`updates`（代理进度）、`messages`（LLM 令牌 + 元数据）或`custom`（任意用户数据）。

## Structured ouput

结构化输出允许代理以特定且可预测的格式返回数据。您无需解析自然语言响应，即可获得以 JSON 对象、Pydantic 模型或数据类形式存在的结构化数据，您的应用程序可以直接使用这些数据。

LangChain[`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent)能够自动处理结构化输出。用户设置所需的结构化输出模式，当模型生成结构化数据时，这些数据会被捕获、验证，并以`'structured_response'`代理状态键的形式返回。

```py
def create_agent(
    ...
    response_format: Union[
        ToolStrategy[StructuredResponseT],
        ProviderStrategy[StructuredResponseT],
        type[StructuredResponseT],
    ]
```

## Middleware

中间件提供一种更严格地控制Agent内部运行机制的方法。中间件的用途包括：

- 通过日志记录、分析和调试来跟踪agent行为。
- 转换提示词、[工具选择](https://docs.langchain.com/oss/python/langchain/middleware/built-in#llm-tool-selector)和输出格式。
- 添加[重试](https://docs.langchain.com/oss/python/langchain/middleware/built-in#tool-retry)、[回退](https://docs.langchain.com/oss/python/langchain/middleware/built-in#model-fallback)和提前终止逻辑。
- 应用[速率限制](https://docs.langchain.com/oss/python/langchain/middleware/built-in#model-call-limit)、防护措施和[个人身份信息检测](https://docs.langchain.com/oss/python/langchain/middleware/built-in#pii-detection)。

通过向以下参数传递中间件来添加它们[`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent)：

```py
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(...),
        HumanInTheLoopMiddleware(...)
    ],
)
```

Agent  Loop

核心Agent循环包括调用模型，让模型选择要执行的工具，然后在不再需要调用任何工具时结束：

![image-20251229180048224](https://ray-blog.oss-cn-hangzhou.aliyuncs.com/images/image-20251229180048224.png)

中间件在每个步骤之前和之后都会暴露钩子：

![image-20251229175613811](https://ray-blog.oss-cn-hangzhou.aliyuncs.com/images/image-20251229175613811.png)

### 内置中间件

LangChain 为常见用例提供预构建的中间件。每个中间件都已准备好投入生产环境，并且可以根据您的具体需求进行配置。

| Middleware                                                   | Description                                  |
| :----------------------------------------------------------- | :------------------------------------------- |
| [Summarization](https://docs.langchain.com/oss/python/langchain/middleware/built-in#summarization) | 当接近会话次数上限时，自动汇总对话历史记录。 |
| [Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/middleware/built-in#human-in-the-loop) | 暂停执行，等待人工审核工具调用。             |
| [Model call limit](https://docs.langchain.com/oss/python/langchain/middleware/built-in#model-call-limit) | 限制模型调用次数，以防止成本过高。           |
| [Tool call limit](https://docs.langchain.com/oss/python/langchain/middleware/built-in#tool-call-limit) | 通过限制调用次数来控制工具执行。             |
| [Model fallback](https://docs.langchain.com/oss/python/langchain/middleware/built-in#model-fallback) | 当主模型发生故障时，自动回退到备用模型。     |
| [PII detection](https://docs.langchain.com/oss/python/langchain/middleware/built-in#pii-detection) | 检测和处理个人身份信息（PII）。              |
| [To-do list](https://docs.langchain.com/oss/python/langchain/middleware/built-in#to-do-list) | 为agent配备任务规划和跟踪功能。              |
| [LLM tool selector](https://docs.langchain.com/oss/python/langchain/middleware/built-in#llm-tool-selector) | 使用 LLM 在调用主模型之前选择相关工具。      |
| [Tool retry](https://docs.langchain.com/oss/python/langchain/middleware/built-in#tool-retry) | 使用指数退避算法自动重试失败的工具调用。     |
| [Model retry](https://docs.langchain.com/oss/python/langchain/middleware/built-in#model-retry) | 使用指数退避算法自动重试失败的模型调用。     |
| [LLM tool emulator](https://docs.langchain.com/oss/python/langchain/middleware/built-in#llm-tool-emulator) | 使用 LLM 模拟工具执行以进行测试。            |
| [Context editing](https://docs.langchain.com/oss/python/langchain/middleware/built-in#context-editing) | 通过精简或清除工具使用情况来管理对话上下文。 |
| [Shell tool](https://docs.langchain.com/oss/python/langchain/middleware/built-in#shell-tool) | 向agent公开持久 shell 会话以执行命令。       |
| [File search](https://docs.langchain.com/oss/python/langchain/middleware/built-in#file-search) | 提供对文件系统文件的 Glob 和 Grep 搜索工具。 |

#### Summarization 总结

当接近消息数量上限时，自动生成对话历史记录摘要，保留最近的消息，同时压缩较早的上下文。摘要功能适用于以下情况：

- 持续时间过长的对话，超出上下文窗口。
- 多轮对话，历史悠久。
- 需要保留完整对话上下文的应用场景。

```py
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[your_weather_tool, your_calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20),
        ),
    ],
)
```

#### Human-in-the-loop   人机交互

在工具调用执行前，暂停代理程序的执行，以便人工审批、编辑或拒绝这些调用。[人机协作机制](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)在以下情况下非常有用：

- 需要人工批准的高风险操作（例如数据库写入、金融交易）。
- 需要人工监督的合规工作流程。
- 长时间的对话，其中人类的反馈会指导智能体。

```py
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver


def read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"

def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"

agent = create_agent(
    model="gpt-4o",
    tools=[your_read_email_tool, your_send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "your_send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "your_read_email_tool": False,
            }
        ),
    ],
)
```

#### Model call limit 模型调用限制

限制模型调用次数以防止无限循环或过高的成本。模型调用次数限制在以下情况下非常有用：

- 防止失控代理发出过多 API 调用。
- 对生产部署实施成本控制。
- 在特定通话预算范围内测试客服人员的行为。

```py
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o",
    checkpointer=InMemorySaver(),  # Required for thread limiting
    tools=[],
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=10,
            run_limit=5,
            exit_behavior="end",
        ),
    ],
)
```

#### Tool call limit 工具调用限制

通过限制工具调用次数来控制代理的执行，可以全局限制所有工具，也可以限制特定工具。工具调用次数限制在以下情况下非常有用：

- 防止过度调用昂贵的外部 API。
- 限制网络搜索或数据库查询。
- 对特定工具的使用频率实施限制。
- 防止代理程序陷入失控循环。

```py
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, database_tool],
    middleware=[
        # Global limit
        ToolCallLimitMiddleware(thread_limit=20, run_limit=10),
        # Tool-specific limit
        ToolCallLimitMiddleware(
            tool_name="search",
            thread_limit=5,
            run_limit=3,
        ),
    ],
)
```

#### Model fallback 模型回退

当主模型失效时，自动回退到备用模型。模型回退功能适用于以下情况：

- 构建能够处理模型故障的弹性代理。
- 通过采用更便宜的型号来优化成本。
- OpenAI、Anthropic 等供应商之间的冗余。

```py
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        ModelFallbackMiddleware(
            "gpt-4o-mini",
            "claude-3-5-sonnet-20241022",
        ),
    ],
)
```

#### PII Detector PII 检测

使用可配置策略检测和处理对话中的个人身份信息 (PII)。PII 检测可用于以下用途：

- 医疗保健和金融应用，需满足合规性要求。
- 需要清理日志的客服人员。
- 任何处理敏感用户数据的应用程序。

```py
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
    ],
)
```

内置PII类型：

- `email`电子邮件地址
- `credit_card`信用卡号码（使用 Luhn 算法验证）
- `ip`IP 地址（已使用标准库验证）
- `mac_address`MAC地址
- `url`URL（包括`http`/`https`和裸 URL）

策略：

- `block`检测到个人身份信息时引发异常
- `redact`：将个人身份信息替换为`[REDACTED_TYPE]`占位符
- `mask`部分屏蔽个人身份信息（例如，`****-****-****-1234`用于信用卡）
- `hash`将 PII 替换为确定性哈希值（例如，`<email_hash:a1b2c3d4>`）

策略选择指南：

| 战略     | 保留身份？   | 最适合                         |
| :------- | :----------- | :----------------------------- |
| `block`  | 不适用       | 完全避免使用个人身份信息       |
| `redact` | 不           | 一般合规性，日志清理           |
| `mask`   | 不           | 人性化易读性，客户服务用户界面 |
| `hash`   | 是的（化名） | 分析、调试                     |

| `apply_to_input`        | 模型调用前是否检查用户消息。**类型：** `bool`**默认：** `True` |
| ----------------------- | ------------------------------------------------------------ |
| `apply_to_output`       | 模型调用后是否检查AI消息。**类型：** `bool`**默认：** `False` |
| `apply_to_tool_results` | 工具执行后是否检查工具结果消息。**类型：** `bool`**默认：** `False` |
| `detector`              | 自定义检测器函数或正则表达式模式。                           |

##### 自定义PII类型

**创建自定义检测器的三种方法：**

1. **正则表达式模式字符串**- 简单模式匹配
2. **自定义功能**- 带有验证功能的复杂检测逻辑

```py
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
import re


# Method 1: Regex pattern string
agent1 = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
        ),
    ],
)

# Method 2: Compiled regex pattern
agent2 = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        PIIMiddleware(
            "phone_number",
            detector=re.compile(r"\+?\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{4}"),
            strategy="mask",
        ),
    ],
)

# Method 3: Custom detector function
def detect_ssn(content: str) -> list[dict[str, str | int]]:
    """Detect SSN with validation.

    Returns a list of dictionaries with 'text', 'start', and 'end' keys.
    """
    import re
    matches = []
    pattern = r"\d{3}-\d{2}-\d{4}"
    for match in re.finditer(pattern, content):
        ssn = match.group(0)
        # Validate: first 3 digits shouldn't be 000, 666, or 900-999
        first_three = int(ssn[:3])
        if first_three not in [0, 666] and not (900 <= first_three <= 999):
            matches.append({
                "text": ssn,
                "start": match.start(),
                "end": match.end(),
            })
    return matches

agent3 = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        PIIMiddleware(
            "ssn",
            detector=detect_ssn,
            strategy="hash",
        ),
    ],
)
```

3. **自定义检测器函数签名：**检测器函数必须接受一个字符串（内容）并返回匹配项：`text`返回一个包含`a`、 `b``start`和`c` 键的字典列表`end`：

### 自定义中间件

通过在agent执行流程的特定点运行钩子来构建自定义中间件。

#### hook 钩子

通过在Agent执行流程的特定点运行钩子来构建自定义中间件。

#### Node-style hooks

按特定执行点顺序运行。用于日志记录、验证和状态更新。**可用挂钩：**

- `before_agent`- 在代理启动之前（每次调用一次）
- `before_model`- 在每次模型调用之前
- `after_model`- 每次模型响应后
- `after_agent`- 代理完成后（每次调用一次）

#### Wrap-style hooks

拦截执行过程，并在调用处理程序时进行控制。可用于重试、缓存和转换。您可以决定处理程序是被调用零次（短路）、一次（正常流程）还是多次（重试逻辑）。**可用挂钩：**

- `wrap_model_call`- 每次模型调用前后
- `wrap_tool_call`- 每次工具调用前后

#### 创建中间件

创建中间件有两种方法

##### 基于装饰器的中间件

适用于单钩子中间件，快速简便。使用装饰器封装各个函数。**可供选择的装饰师：****节点式：**

- [`@before_agent`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.before_agent)- 在代理启动前运行（每次调用运行一次）
- [`@before_model`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.before_model)- 在每次模型调用之前运行
- [`@after_model`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.after_model)- 在每次模型响应后运行
- [`@after_agent`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.after_agent)- 在代理程序完成后运行（每次调用一次）

**环绕式：**

- [`@wrap_model_call`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.wrap_model_call)- 使用自定义逻辑包装每个模型调用
- [`@wrap_tool_call`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.wrap_tool_call)- 使用自定义逻辑包装每个工具调用

**方便：**

- [`@dynamic_prompt`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.dynamic_prompt)- 生成动态系统提示

##### 基于类的中间件

对于具有多个钩子或配置的复杂中间件，类的功能更强大。当需要为同一个钩子定义同步和异步实现，或者想要在单个中间件中组合多个钩子时，请使用类。

# 高级用法

## Guardrails  护栏

为您的Agent实施安全检查和内容过滤。

防护机制通过在Agent执行的关键节点验证和过滤内容，帮助您构建安全合规的 AI 应用。它们可以检测敏感信息、强制执行内容策略、验证输出，并在不安全行为造成问题之前加以阻止。常见应用场景包括：

- 防止个人身份信息泄露
- 检测和阻止提示注入攻击
- 屏蔽不当或有害内容
- 执行业务规则和合规要求
- 验证输出质量和准确性

您可以使用[中间件](https://docs.langchain.com/oss/python/langchain/middleware)来实现防护措施，在关键点拦截执行——在代理启动之前、完成之后，或者在模型和工具调用前后。

### 内置护栏

PII检测

人机交互

### 定制护栏

对于更复杂的防护措施，您可以创建自定义中间件，使其在Agent执行之前或之后运行。这样，您就可以完全控制验证逻辑、内容过滤和安全检查。

在Agent护栏之前

使用“before agent”钩子在每次调用开始时验证请求一次。这对于会话级检查（例如身份验证、速率限制或在任何处理开始之前阻止不当请求）非常有用。

```py
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime

class ContentFilterMiddleware(AgentMiddleware):
    """Deterministic guardrail: Block requests containing banned keywords."""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # Get the first user message
        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None

        content = first_message.content.lower()

        # Check for banned keywords
        for keyword in self.banned_keywords:
            if keyword in content:
                # Block execution before any processing
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": "I cannot process requests containing inappropriate content. Please rephrase your request."
                    }],
                    "jump_to": "end"
                }

        return None

# Use the custom guardrail
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, calculator_tool],
    middleware=[
        ContentFilterMiddleware(
            banned_keywords=["hack", "exploit", "malware"]
        ),
    ],
)

# This request will be blocked before any processing
result = agent.invoke({
    "messages": [{"role": "user", "content": "How do I hack into a database?"}]
})
```

Agent护栏之后

使用“agent后”钩子在返回给用户之前对最终输出进行一次验证。这对于基于模型的安全检查、质量验证或对完整代理响应进行最终合规性扫描非常有用。

```py
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime
from langchain.messages import AIMessage
from langchain.chat_models import init_chat_model
from typing import Any

class SafetyGuardrailMiddleware(AgentMiddleware):
    """Model-based guardrail: Use an LLM to evaluate response safety."""

    def __init__(self):
        super().__init__()
        self.safety_model = init_chat_model("gpt-4o-mini")

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # Get the final AI response
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        # Use a model to evaluate safety
        safety_prompt = f"""Evaluate if this response is safe and appropriate.
        Respond with only 'SAFE' or 'UNSAFE'.

        Response: {last_message.content}"""

        result = self.safety_model.invoke([{"role": "user", "content": safety_prompt}])

        if "UNSAFE" in result.content:
            last_message.content = "I cannot provide that response. Please rephrase your request."

        return None

# Use the safety guardrail
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, calculator_tool],
    middleware=[SafetyGuardrailMiddleware()],
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "How do I make explosives?"}]
})
```

### 组合多个护栏

您可以通过将多个防护措施添加到中间件数组中来堆叠它们。它们按顺序执行，从而允许您构建分层保护：

```py
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, send_email_tool],
    middleware=[
        # Layer 1: Deterministic input filter (before agent)
        ContentFilterMiddleware(banned_keywords=["hack", "exploit"]),

        # Layer 2: PII protection (before and after model)
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("email", strategy="redact", apply_to_output=True),

        # Layer 3: Human approval for sensitive tools
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),

        # Layer 4: Model-based safety check (after agent)
        SafetyGuardrailMiddleware(),
    ],
)
```

## Runtime 运行时

LangChain[`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent)底层运行在LangGraph的运行时环境中。LangGraph 公开了一个[`Runtime`](https://reference.langchain.com/python/langgraph/runtime/#langgraph.runtime.Runtime)包含以下信息的对象：

1. **上下文**：静态信息，例如用户 ID、数据库连接或其他代理调用依赖项。
2. **存储**：用于[长期记忆的](https://docs.langchain.com/oss/python/langchain/long-term-memory)[BaseStore实例](https://reference.langchain.com/python/langgraph/store/#langgraph.store.base.BaseStore)
3. **流写入器**`"custom"`：用于通过流模式传输信息的对象

[您可以在工具](https://docs.langchain.com/oss/python/langchain/runtime#inside-tools)和[中间件](https://docs.langchain.com/oss/python/langchain/runtime#inside-middleware)中访问运行时信息。

### Access 使用权

使用 创建一个代理时[`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent)，您可以指定来定义存储在代理中`context_schema`的 的结构。`context`[`Runtime`](https://reference.langchain.com/python/langgraph/runtime/#langgraph.runtime.Runtime)调用代理时，请传递`context`包含运行相关配置的参数：

```py
from dataclasses import dataclass

from langchain.agents import create_agent


@dataclass
class Context:
    user_name: str

agent = create_agent(
    model="gpt-5-nano",
    tools=[...],
    context_schema=Context
)

agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    context=Context(user_name="John Smith")
)
```

### Inside tools 内部工具

您可以通过访问工具内部的运行时信息来执行以下操作：

- 获取上下文
- 读取或写入长期记忆
- 写入自[定义流](https://docs.langchain.com/oss/python/langchain/streaming#custom-updates)（例如，工具进度/更新）

使用该`ToolRuntime`参数可以访问[`Runtime`](https://reference.langchain.com/python/langgraph/runtime/#langgraph.runtime.Runtime)工具内部的对象。

```py
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@dataclass
class Context:
    user_id: str

@tool
def fetch_user_email_preferences(runtime: ToolRuntime[Context]) -> str:
    """Fetch the user's email preferences from the store."""
    user_id = runtime.context.user_id

    preferences: str = "The user prefers you to write a brief and polite email."
    if runtime.store:
        if memory := runtime.store.get(("users",), user_id):
            preferences = memory.value["preferences"]

    return preferences
```

### Inside middleware 中间件内部

您可以访问中间件中的运行时信息，以根据用户上下文创建动态提示、修改消息或控制代理行为。用于在中间件装饰器中`request.runtime`访问[`Runtime`](https://reference.langchain.com/python/langgraph/runtime/#langgraph.runtime.Runtime)对象。运行时对象可通过[`ModelRequest`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.ModelRequest)传递给中间件函数的参数访问。

```py
from dataclasses import dataclass

from langchain.messages import AnyMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import dynamic_prompt, ModelRequest, before_model, after_model
from langgraph.runtime import Runtime


@dataclass
class Context:
    user_name: str

# Dynamic prompts
@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context.user_name
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    return system_prompt

# Before model hook
@before_model
def log_before_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    print(f"Processing request for user: {runtime.context.user_name}")
    return None

# After model hook
@after_model
def log_after_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    print(f"Completed request for user: {runtime.context.user_name}")
    return None

agent = create_agent(
    model="gpt-5-nano",
    tools=[...],
    middleware=[dynamic_system_prompt, log_before_model, log_after_model],
    context_schema=Context
)

agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    context=Context(user_name="John Smith")
)
```

## Context engineering in agents Agent中的上下文工程

构建Agent（或任何LLM应用程序）的难点在于如何确保其可靠性。虽然它们在原型中可能有效，但在实际应用场景中往往会失败。

**上下文工程**是指以正确的格式提供正确的信息和工具，以便语言学习模型（LLM）能够完成任务。这是人工智能工程师的首要任务。缺乏“正确”的上下文是构建更可靠智能体的最大障碍，而 LangChain 的智能体抽象设计旨在促进上下文工程。

### 你能控制的

要构建可靠的代理，你需要控制代理循环中每一步发生的事情，以及步骤之间发生的事情。

| 上下文类型                                                   | 你能掌控什么                                                 | 短暂的或持续的 |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :------------- |
| **[模型上下文](https://docs.langchain.com/oss/python/langchain/context-engineering#model-context)** | 模型调用中包含哪些内容（指令、消息历史记录、工具、响应格式） | 瞬态           |
| **[工具上下文](https://docs.langchain.com/oss/python/langchain/context-engineering#tool-context)** | 哪些工具可以访问和生成（对状态、存储、运行时上下文的读/写操作） | 执着的         |
| **[生命周期背景](https://docs.langchain.com/oss/python/langchain/context-engineering#life-cycle-context)** | 模型调用和工具调用之间发生了什么（摘要、防护措施、日志记录等） | 执着的         |

### 工作原理

LangChain[中间件](https://docs.langchain.com/oss/python/langchain/middleware)是底层机制，它使使用 LangChain 的开发人员能够实际进行上下文工程。中间件允许您接入代理生命周期中的任何步骤，并：

- 更新上下文
- 跳转到代理生命周期中的其他步骤

在本指南中，您将经常看到中间件 API 被用作实现上下文工程的手段。

### 模型上下文

控制每次模型调用的内容——指令、可用工具、使用的模型以及输出格式。这些决策直接影响可靠性和成本。

- 系统提示
- 消息
- 工具
- 模型
- 回复格式

所有这些类型的模型上下文都可以从**状态**（短期记忆）、**存储**（长期记忆）或**运行时上下文**（静态配置）中获取信息。

#### 系统提示

##### 状态（短期记忆）

```py
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    # request.messages is a shortcut for request.state["messages"]
    message_count = len(request.messages)

    base = "You are a helpful assistant."

    if message_count > 10:
        base += "\nThis is a long conversation - be extra concise."

    return base

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[state_aware_prompt]
)
```

##### 存储（长期记忆）

```py
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.user_id

    # Read from Store: get user preferences
    store = request.runtime.store
    user_prefs = store.get(("preferences",), user_id)

    base = "You are a helpful assistant."

    if user_prefs:
        style = user_prefs.value.get("communication_style", "balanced")
        base += f"\nUser prefers {style} responses."

    return base

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[store_aware_prompt],
    context_schema=Context,
    store=InMemoryStore()
)
```

##### 运行时上下文（静态配置）

```py
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dataclass
class Context:
    user_role: str
    deployment_env: str

@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    # Read from Runtime Context: user role and environment
    user_role = request.runtime.context.user_role
    env = request.runtime.context.deployment_env

    base = "You are a helpful assistant."

    if user_role == "admin":
        base += "\nYou have admin access. You can perform all operations."
    elif user_role == "viewer":
        base += "\nYou have read-only access. Guide users to read operations only."

    if env == "production":
        base += "\nBe extra careful with any data modifications."

    return base

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[context_aware_prompt],
    context_schema=Context
)
```

#### 消息

状态（短期记忆）

存储（长期记忆）

运行时上下文（静态配置）

#### 工具

#### 模型

#### 回复格式

## Model Context Protocol (MCP) 模型上下文协议

[模型上下文协议 (MCP)](https://modelcontextprotocol.io/introduction)是一种开放协议，它规范了应用程序如何向语言学习模型 (LLM) 提供工具和上下文。LangChain 代理可以使用[`langchain-mcp-adapters`](https://github.com/langchain-ai/langchain-mcp-adapters)库来使用 MCP 服务器上定义的工具。

### 快速入门

安装`langchain-mcp-adapters`库：

```shell
pip install langchain-mcp-adapters

uv add langchain-mcp-adapters
```

​

访问多个 MCP 服务器

```py
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


client = MultiServerMCPClient(
    {
        "math": {
            "transport": "stdio",  # Local subprocess communication
            "command": "python",
            # Absolute path to your math_server.py file
            "args": ["/path/to/math_server.py"],
        },
        "weather": {
            "transport": "http",  # HTTP-based remote server
            # Ensure you start your weather server on port 8000
            "url": "http://localhost:8000/mcp",
        }
    }
)

tools = await client.get_tools()
agent = create_agent(
    "claude-sonnet-4-5-20250929",
    tools
)
math_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
)
weather_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
)
```

### 自定义服务器

要创建自定义 MCP 服务器，请使用[FastMCP](https://gofastmcp.com/getting-started/welcome)库：

```shell
pip install fastmcp
uv add fastmcp
```

标准输入输出

```shell
from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

可流式HTTP传输

```py
from fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return "It's always sunny in New York"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

### Transports  传输

MCP 支持不同的客户端-服务器通信传输机制。

#### HTTP

该`http`传输层（也称为HTTP传输层`streamable-http`）使用HTTP请求进行客户端与服务器之间的通信。

##### 传递头部

通过 HTTP 连接到 MCP 服务器时，您可以使用`headers`连接配置中的字段包含自定义标头（例如，用于身份验证或跟踪）。此功能`sse`受 MCP 规范中已弃用的`streamable_http`协议支持。

##### 验证

该`langchain-mcp-adapters`库底层使用了官方的[MCP SDK](https://github.com/modelcontextprotocol/python-sdk)，允许您通过实现`httpx.Auth`接口来提供自定义身份验证机制。

#### 标准排版

客户端以子进程方式启动服务器，并通过标准输入/输出进行通信。最适合本地工具和简单配置。

### 核心功能

工具、资源、提示

#### 工具

[工具](https://modelcontextprotocol.io/docs/concepts/tools)允许 MCP 服务器公开可执行函数，供 LLM 调用以执行操作，例如查询数据库、调用 API 或与外部系统交互。LangChain 将 MCP 工具转换为 LangChain[工具](https://docs.langchain.com/oss/python/langchain/tools)，使其可以直接在任何 LangChain 代理或工作流中使用。

##### 加载工具

用于`client.get_tools()`从 MCP 服务器检索工具并将其传递给您的Agent：

```py
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient({...})
tools = await client.get_tools()
agent = create_agent("claude-sonnet-4-5-20250929", tools)
```

##### 结构化内容

MCP 工具除了返回人类可读的文本响应外，还可以返回[结构化内容](https://modelcontextprotocol.io/specification/2025-03-26/server/tools#structured-content)。当工具除了向模型显示文本外，还需要返回机器可解析的数据（例如 JSON）时，此功能非常有用。当 MCP 工具返回数据时`structuredContent`，适配器会将其包装在一个对象中[`MCPToolArtifact`](https://docs.langchain.com/docs/reference/langchain-mcp-adapters#MCPToolArtifact)，并将其作为工具的工件返回。您可以使用`artifact`对象上的字段访问此数据`ToolMessage`。您还可以使用[拦截器](https://docs.langchain.com/oss/python/langchain/mcp#tool-interceptors)自动处理或转换结构化内容。

##### 多模态工具内容

MCP 工具可以在响应中返回[多模态内容（图像、文本等）。当 MCP 服务器返回包含多个部分（例如文本和图像）的内容时，适配器会将其转换为 LangChain 的](https://modelcontextprotocol.io/specification/2025-03-26/server/tools#tool-result)[标准内容块](https://docs.langchain.com/oss/python/langchain/messages#standard-content-blocks)`content_blocks`

#### 资源

[资源](https://modelcontextprotocol.io/docs/concepts/resources)允许 MCP 服务器公开可供客户端读取的数据，例如文件、数据库记录或 API 响应。LangChain 将 MCP 资源转换为[Blob](https://docs.langchain.com/docs/reference/langchain-core/documents#Blob)对象，Blob 对象为处理文本和二进制内容提供了一个统一的接口。

##### 正在加载资源

用于`client.get_resources()`从 MCP 服务器加载资源：

```py
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({...})

# Load all resources from a server
blobs = await client.get_resources("server_name")

# Or load specific resources by URI
blobs = await client.get_resources("server_name", uris=["file:///path/to/file.txt"])

for blob in blobs:
    print(f"URI: {blob.metadata['uri']}, MIME type: {blob.mimetype}")
    print(blob.as_string())  # For text content
```

#### 提示

[提示功能](https://modelcontextprotocol.io/docs/concepts/prompts)允许 MCP 服务器公开可重用的提示模板，供客户端检索和使用。LangChain 将 MCP 提示转换为[消息](https://docs.langchain.com/docs/concepts/messages)，使其易于集成到基于聊天的工作流程中。

##### 加载提示

用于`client.get_prompt()`从 MCP 服务器加载提示符：

```py
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({...})

# Load a prompt by name
messages = await client.get_prompt("server_name", "summarize")

# Load a prompt with arguments
messages = await client.get_prompt(
    "server_name",
    "code_review",
    arguments={"language": "python", "focus": "security"}
)

# Use the messages in your workflow
for message in messages:
    print(f"{message.type}: {message.content}")
```
