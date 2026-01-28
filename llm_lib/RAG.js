import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import { ChatPromptTemplate } from "@langchain/core/prompts";

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";

export { Document };

/**
 * @template T
 */

export class RAG {
    /** @type {ChatPromptTemplate} */
    #promptTemplate

    /** @type {Ollama} */
    #llm

    /** @type {MemoryVectorStore} */
    #vectorStore

    /** @type {Array<Document>} */
    #docs

    /**
     * 
     * @param {string} host 
     * @param {string} llmModelName 
     * @param {string} embeddingsModelName 
     * @param {string} template
     */
    constructor(host = "localhost", llmModelName = "gemma3:4b", embeddingsModelName = "nomic-embed-text", template) {
        this.#docs = [];

        this.#llm = new Ollama({
            baseUrl: host,
            model: llmModelName,
            temperature: 0,
            maxRetries: 2,
            keepAlive: "50000m"
        });

        let embeddings = new OllamaEmbeddings({
            baseUrl: host,
            model: embeddingsModelName,
            temperature: 0,
            maxRetries: 2,
            keepAlive: "50000m"
        });

        this.#vectorStore = new MemoryVectorStore(embeddings);

        this.#promptTemplate = ChatPromptTemplate.fromMessages([
            ["system", template]
        ]);
    }

    /**
     * 
     * @param {Document} doc 
     * @returns {RAG<T>}
     */
    addDocument(doc) {
        this.#docs.push(doc);

        return this
    }

    /**
     * 
     * @param {Array<Document>} docs 
     * @returns {RAG<T>}
     */
    addDocuments(docs) {
        this.#docs.push(...docs);

        return this
    }

    /**
     * 
     * @param {number} chunkSize 
     * @param {number} chunkOverlap 
     * @returns {Promise<RAG<T>>}
     */
    async genericTextDocumentSplitter(chunkSize = 1000, chunkOverlap = 200) {
        const splitter = new RecursiveCharacterTextSplitter({ chunkSize, chunkOverlap });
        this.#docs = await splitter.splitDocuments(this.#docs);

        return this;
    }

    /**
     * Aplica las acciones de addDocument, addDocuments y genericTextDocumentSplitter hechas hasta ahora
     */
    async saveContext() {
        await this.#vectorStore.addDocuments(this.#docs);
        this.#docs = [];
    }

    /**
     * 
     * @param {T} templateParams 
     * @returns {Promise<string>}
     */
    async ask(templateParams) {
        let retrievedDocs = await this.#vectorStore.similaritySearchWithScore(templateParams.question, 50);
        retrievedDocs = retrievedDocs.map(v => v[0]);

        const docsContent = retrievedDocs.map((doc) => doc.pageContent).join("\n");

        const messages = await this.#promptTemplate.invoke({
            context: docsContent,
            ...templateParams
        });

        return await this.#llm.invoke(messages);
    }

    /**
     * WIP
     * @param {WebSocket} ws 
     * @param {T} templateParams 
     * @returns {string}
    */
    async askStream(ws, templateParams) {
        console.log("asked", templateParams.question);

        let retrievedDocs = await this.#vectorStore.similaritySearchWithScore(templateParams.question, 40);
        console.log(retrievedDocs.map(v => v[0].pageContent + " | " + v[1].toFixed(2)));
        retrievedDocs = retrievedDocs.map(v => v[0]);

        const docsContent = retrievedDocs.map((doc) => doc.pageContent).join("\n");

        const messages = await this.#promptTemplate.invoke({
            context: docsContent,
            ...templateParams
        });

        let out = "";

        for await (const event of await this.#llm.streamEvents(messages, { version: "v2" })) {
            if(event.event === "on_llm_stream") {
                ws.send(event.data.chunk.text);
                out += event.data.chunk.text;
            }
        }

        return out;
    }

}