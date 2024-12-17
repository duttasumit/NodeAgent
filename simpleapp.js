
/*Steps:
1. Create the graph state StateAnnotation using Annotation.root, Annotation, messageStateReducer
2. Create external tool using tool(asyc ({query}) => {}, {name, description, schema})
3. Create tools array
4. Create ToolNode with the tools array: new ToolNode(tools)
5. Instatiate model and bind with tools (bindTools())
6. Create a callModel function for the agent in StateGraph, that include model.invoke(messages) logic
7. Create toolconditional function if required
8. Create Graph using StateGraph(StateAnnotation)
9. Add nodes first, "agent", "tool"
10. Add edges, normal and conditional (addConditionalEdges)
11. Compile the graph 
12. To visualize the graph:
    - getGraphAsync(), drawMerMaidPNG() - this returns blob
    - convert blob to ArrayBuffer - blob.arrayBuffer()
    - convert arrayBuffer to Buffer - Buffer.from(arrayBuffer)
    - fs.writeFileSync(filename, buffer)
13. Invoke to get the response.
*/

import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatGroq } from "@langchain/groq";
import { StateGraph } from "@langchain/langgraph";
import { MemorySaver, Annotation, messagesStateReducer } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import 'dotenv/config'
// import * as fs from 'fs'


// Define the graph state.
// Define how updates from each node merge in the graph state: StateAnnotation (Annotation)
const StateAnnotation = Annotation.Root({
    messages: Annotation({
        // `messagesStateReducer` function defines how `messages` state key should be updated
        reducer: messagesStateReducer
    })
})


// Define the tools for the agent to use. (External tool)
const weatherTool = tool(
    async ({ query }) => {
        if (query.toLowerCase().includes("sf") || query.toLowerCase().includes("san francisco")) {
            return "It's 60 degrees and foggy."
        }
        return "It's 90 degrees and sunny."
    },
    {
        name: "weather",
        description: "Call to get the current weather for a location.",
        schema: z.object({
            query: z.string().describe("The query to use in the search.")
        })
    }
)
const tools = [weatherTool]


// Instantiate the ToolNode object.
const toolNode = new ToolNode(tools)


// Instantiate the model and bind with the tools.
const model = new ChatGroq({
    model: 'llama3-8b-8192',
    temperature: 0
}).bindTools(tools)


// Define the function(conditional) to determine whether to continue or not
// We can extract the state typing via StateAnnotation.State
function shouldContinue(state) {
    const messages = state.messages
    const lastMessage = messages[messages.length - 1]

    // LLM makes the tool call, route to the "tools" node
    if (lastMessage.tool_calls?.length) {
        return 'tools'
    }
    return "__end__"
}


// Define the function that calls the model.
async function callModel(state) {
    const messages = state.messages
    const response = await model.invoke(messages)
    return { messages: [response] }
}


// Define a new graph.
const workflow = new StateGraph(StateAnnotation)
    .addNode("agent", callModel)
    .addNode("tools", toolNode)
    .addEdge("__start__", "agent")
    .addConditionalEdges("agent", shouldContinue)
    .addEdge("tools", "agent")
    .addEdge("agent", "__end__")


// Initialize memory to persist state between graph runs
// const checkpointer = new MemorySaver()


// Complie it.
// This compiles it into Langchain Runnable.
const app = workflow.compile()


// const graph = await app.getGraphAsync();
// const blob = await graph.drawMermaidPng();

// // Convert Blob to ArrayBuffer
// const arrayBuffer = await blob.arrayBuffer();

// // Convert ArrayBuffer to Buffer
// const buffer = Buffer.from(arrayBuffer);

// // Write the Buffer to a file
// fs.writeFileSync("graph.jpeg", buffer);
// console.log("Graph saved as graph.jpeg");


// Use the Runnable.
const finalState = await app.invoke(
    { messages: [new HumanMessage("What is the weather in sf")]},
    { configurable: { thread_id: '11'}}
)

console.log(finalState.messages[finalState.messages.length - 1].content)