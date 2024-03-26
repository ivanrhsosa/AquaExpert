# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

# OpenAI Chat completion
import os
import getpass
from openai import AsyncOpenAI

import chainlit as cl 
from chainlit.prompt import Prompt, PromptMessage 
from chainlit.playground.providers import ChatOpenAI

from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
from uuid import uuid4

from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolExecutor

import operator
from typing import Annotated, Sequence, TypedDict

import json

from langchain import hub
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolInvocation
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

import pprint

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"AIE1 - AquaExpert - {uuid4().hex[0:8]}"



@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    vr_vector_store = FAISS.load_local("faiss_visitreports_index", embeddings, allow_dangerous_deserialization=True)
    retriever_visitreports = vr_vector_store.as_retriever()

    ct_vector_store = FAISS.load_local("faiss_coolingtower_index", embeddings, allow_dangerous_deserialization=True)
    retriever_coolingtower = ct_vector_store.as_retriever()

    cl_vector_store = FAISS.load_local("faiss_closedloop_index", embeddings, allow_dangerous_deserialization=True)
    retriever_closedloop = cl_vector_store.as_retriever()

    customer_reports = create_retriever_tool(
      retriever_visitreports,
      "retrieve_reports_info",
      "Search and return specific information stored in cooling tower and closed-loop system service reports. "
      "This includes but is not limited to chemical treatment details, operational parameters, maintenance activities, "
      "system performance data, and recommendations for adjustments. It allows users to query past records for "
      "conductivity levels, pH balances, inhibitor dosages, corrosion and scaling indices, microbial counts, and more. "
      "The retriever can also provide historical trends, equipment status updates, and any flagged issues from the visit logs. "
      "Utilize this tool to gain insights into water quality, chemical balances, and equipment health as reported by service technicians."
)

    cooling_tower_procedures = create_retriever_tool(
      retriever_coolingtower,
      "retrieve_coolingtower_procedures",
      "Search and return specific information stored in the database of cooling tower procedures. "
      "This tool is adept at extracting detailed procedural documentation, inspection guidelines, "
      "and maintenance protocols for cooling towers. Users can retrieve comprehensive steps for "
      "routine checks and complex maintenance tasks, including but not limited to film fill inspections, "
      "distribution deck examinations, general structural assessments, cold water basin inspections, "
      "chemical descaling processes, and rust removal techniques. It serves as an indispensable resource for "
      "ensuring adherence to industry best practices and maintaining the operational integrity of cooling tower systems. "
)

    closed_loop_procedures = create_retriever_tool(
        retriever_closedloop,
        "retrieve_closedloop_procedures",
        "Search and return specific information stored in documentation and reports relevant to closed-loop system maintenance and treatment procedures. "
        "This tool is designed to provide access to a comprehensive set of guidelines and best practices for maintaining closed-loop systems, including but not limited to disinfection processes, descaling operations, and iron deposit removal. "
        "Whether you're looking for step-by-step instructions for small closed loop disinfection or methods for utilizing Magcare 300 in descaling and deposit removal, this retriever tool can efficiently locate and present the necessary procedures from the vector database. "
    )

    tools = [customer_reports, cooling_tower_procedures, closed_loop_procedures]

    tool_executor = ToolExecutor(tools)

    class AgentState(TypedDict):
      messages: Annotated[Sequence[BaseMessage], operator.add]

    def should_retrieve(state):
      """
      Decides whether the agent should retrieve more information or end the process.

      This function checks the last message in the state for a function call. If a function call is
      present, the process continues to retrieve information. Otherwise, it ends the process.

      Args:
          state (messages): The current state

      Returns:
          str: A decision to either "continue" the retrieval process or "end" it
      """

      print("---DECIDE TO RETRIEVE---")
      messages = state["messages"]
      last_message = messages[-1]

      # If there is no function call, then we finish
      if "function_call" not in last_message.additional_kwargs:
          print("---DECISION: DO NOT RETRIEVE / DONE---")
          return "end"
      # Otherwise there is a function call, so we continue
      else:
          print("---DECISION: RETRIEVE---")
          return "continue"


    def grade_documents(state):
      """
      Determines whether the retrieved documents are relevant to the question.

      Args:
          state (messages): The current state

      Returns:
          str: A decision for whether the documents are relevant or not
      """

      print("---CHECK RELEVANCE---")

      # Data model
      class grade(BaseModel):
          """Binary score for relevance check."""

          binary_score: str = Field(description="Relevance score 'yes' or 'no'")

      # LLM
      model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

      # Tool
      grade_tool_oai = convert_to_openai_tool(grade)

      # LLM with tool and enforce invocation
      llm_with_tool = model.bind(
          tools=[convert_to_openai_tool(grade_tool_oai)],
          tool_choice={"type": "function", "function": {"name": "grade"}},
      )

      # Parser
      parser_tool = PydanticToolsParser(tools=[grade])

      # Prompt
      prompt = PromptTemplate(
          template="""You are a grader assessing relevance of a retrieved document to a user question. \n
          Here is the retrieved document: \n\n {context} \n\n
          Here is the user question: {question} \n
          If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
          Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
          input_variables=["context", "question"],
      )

      # Chain
      chain = prompt | llm_with_tool | parser_tool

      messages = state["messages"]
      last_message = messages[-1]

      question = messages[0].content
      docs = last_message.content

      score = chain.invoke(
          {"question": question,
          "context": docs}
      )

      grade = score[0].binary_score

      if grade == "yes":
          print("---DECISION: DOCS RELEVANT---")
          return "yes"

      else:
          print("---DECISION: DOCS NOT RELEVANT---")
          print(grade)
          return "no"


  ### Nodes


    def agent(state):
      """
      Invokes the agent model to generate a response based on the current state. Given
      the question, it will decide to retrieve using the retriever tool, or simply end.

      Args:
          state (messages): The current state

      Returns:
          dict: The updated state with the agent response apended to messages
      """
      print("---CALL AGENT---")
      messages = state["messages"]
      model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-0125-preview")
      functions = [format_tool_to_openai_function(t) for t in tools]
      model = model.bind_functions(functions)
      response = model.invoke(messages)
      # We return a list, because this will get added to the existing list
      return {"messages": [response]}

    def retrieve(state):
      """
      Uses tool to execute retrieval.

      Args:
          state (messages): The current state

      Returns:
          dict: The updated state with retrieved docs
      """
      print("---EXECUTE RETRIEVAL---")
      messages = state["messages"]
      # Based on the continue condition
      # we know the last message involves a function call
      last_message = messages[-1]
      # We construct an ToolInvocation from the function_call
      action = ToolInvocation(
          tool=last_message.additional_kwargs["function_call"]["name"],
          tool_input=json.loads(
              last_message.additional_kwargs["function_call"]["arguments"]
          ),
      )
      print("Retrieve Action: ", action)
      # We call the tool_executor and get back a response
      response = tool_executor.invoke(action)
      function_message = FunctionMessage(content=str(response), name=action.tool)

      # We return a list, because this will get added to the existing list
      return {"messages": [function_message]}

    def rewrite(state):
      """
      Transform the query to produce a better question.

      Args:
          state (messages): The current state

      Returns:
          dict: The updated state with re-phrased question
      """

      print("---TRANSFORM QUERY---")
      messages = state["messages"]
      question = messages[0].content

      msg = [HumanMessage(
          content=f""" \n
      Look at the input and try to reason about the underlying semantic intent / meaning. \n
      Here is the initial question:
      \n ------- \n
      {question}
      \n ------- \n
      Formulate an improved question: """,
      )]

      # Grader
      model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
      response = model.invoke(msg)
      return {"messages": [response]}

    def generate(state):
      """
      Generate answer

      Args:
          state (messages): The current state

      Returns:
          dict: The updated state with re-phrased question
      """
      print("---GENERATE---")
      messages = state["messages"]
      question = messages[0].content
      last_message = messages[-1]

      question = messages[0].content
      docs = last_message.content

      # Prompt
      prompt = hub.pull("rlm/rag-prompt")

      # LLM
      llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0, streaming=True)

      # Post-processing
      def format_docs(docs):
          return "\n\n".join(doc.page_content for doc in docs)

      # Chain
      rag_chain = prompt | llm | StrOutputParser()

      # Run
      response = rag_chain.invoke({"context": docs, "question": question})
      return {"messages": [response]}

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the nodes we will cycle between
    workflow.add_node("agent", agent)  # agent
    workflow.add_node("retrieve", retrieve)  # retrieval
    workflow.add_node("rewrite", rewrite)  # retrieval
    workflow.add_node("generate", generate)  # retrieval

    # Call agent node to decide to retrieve or not
    workflow.set_entry_point("agent")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "agent",
        # Assess agent decision
        should_retrieve,
        {
            # Call tool node
            "continue": "retrieve",
            "end": END,
        },
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents,
        {
            "yes": "generate",
            "no": "rewrite",
        },
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    # Compile
    app = workflow.compile()

    def convert_inputs(input_object):
      return {"messages" : [HumanMessage(content=input_object["question"])]}

    def parse_output(input_state):
      return input_state["messages"][-1]

    agent_chain = convert_inputs | app | parse_output

    cl.user_session.set("agent_chain", agent_chain)


@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    agent_chain = cl.user_session.get("agent_chain")

    print(message.content)

    msg = cl.Message(content="")

    result = agent_chain.invoke({"question" : message.content})

    result_content = result["response"].content
    print(result_content)
    msg.content = result_content

    # Send and close the message stream
    await msg.send()
