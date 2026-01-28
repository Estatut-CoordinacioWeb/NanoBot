import { VectorDBQAChain } from "langchain/chains";
import { Document } from "langchain/document";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import ollama from "ollama"

import { AIMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

// Create custom model
// Check: https://github.com/ollama/ollama/blob/main/docs/modelfile.md#instructions
//   await ollama.create({ model: LLM_MODEL_NAME, path: "./model/categorizer-en.mf" });


export class MemoryService {
    constructor(url = "localhost", model = "llama3.2") {
        // specify LLM model
        this.llmModel = new ChatOllama({
            baseUrl: url,
            model: model,
            temperature: 0,
            maxRetries: 2,
            keepAlive: "50000000m"
        });

        // specify embeddings model
        this.embeddingsModel = new OllamaEmbeddings({
            baseUrl: url,
            model: "nomic-embed-text",
            maxRetries: 2,
            keepAlive: "50000000m"
        });

        // create vector store by combining OpenSearch store with the embeddings model
        this.vectorStore = new MemoryVectorStore(this.embeddingsModel);

        // combine the LLM model and the vector store to get a chain
        this.chain = VectorDBQAChain.fromLLM(this.llmModel, this.vectorStore, {
            k: 10,
            returnSourceDocuments: true,
        });

        // this.chain = this.llmModel.pipe(this.vectorStore);

    }

    async storeMemory(memory) {

        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 500,
            chunkOverlap: 50
        });

        // Se parten los documentos para que la llm pueda guardar todos los contextos a la vez
        let rdocs = [...(await textSplitter.splitDocuments([
            new Document({ pageContent: memory })
        ])
        )]

        console.log("Splitted document in", memory.length, "chunks");

        await this.vectorStore.addDocuments(rdocs);
    }

    async storeMemoryJSON(memory) {

        const textSplitter = RecursiveCharacterTextSplitter.fromLanguage("js", {
            chunkSize: 500,
            chunkOverlap: 0
        });

        // Se parten los documentos para que la llm pueda guardar todos los contextos a la vez
        let rdocs = [...(await textSplitter.splitDocuments([
            new Document({ pageContent: memory })
        ])
        )]

        console.log("Splitted document in", memory.length, "chunks");

        await this.vectorStore.addDocuments(rdocs);
    }

    /**
     * 
     * @param {string} query 
     * @returns {Promise<{text: string, sourceDocuments: Array<Document>}>}
     */
    async getRelevantMemory(query) {        
        const response = await this.chain.invoke({ query });
        
        // query = JSON.stringify(query);

        // const response = await this.chain.invoke({ query });

        return response;
    }
}