import { MemoryService } from './llm_lib/_leagacy_memoryService.js';

import { WebSocketServer } from 'ws';
import express from "express";
import fs from "fs";

import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { Document, RAG } from './llm_lib/RAG.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);


const app = express();
const port = 3039;
const WSPort = 3040;

app.use(express.urlencoded({ extended: true }));
app.use(express.json({ limit: "100mb" }));


app.get("*", (req, res) => {
  if (req.path === "/") res.sendFile(__dirname + "/index.html");
  else res.sendFile(__dirname + req.path);
});

app.post("/ask", async (req, res) => {
  /** @type {Array<Message>} */
  const msgs = req.body.messages;

  const response = await rag.ask({ question: msgs });

  res.send({ r: response });
});

/**
 * @typedef {object} RAGTemplate
 * @property {string} context
 * @property {string} question
 * @property {string} history
 * @property {string} hour
 */

/** @type {RAG<RAGTemplate>} */
let rag = new RAG("localhost", "gemma3:4b", "embeddinggemma", fs.readFileSync("./llm_lib/model_rules/nanobot.template").toString());

rag.addDocuments(fs.readFileSync("./llm_lib/data/organigrama2526.txt").toString().split("\n\n").map(v => new Document({ pageContent: v })));
rag.addDocuments(fs.readFileSync("./llm_lib/data/urls2526.txt").toString().split("\n\n").map(v => new Document({ pageContent: v })));
rag.addDocuments(fs.readFileSync("./llm_lib/data/departaments2526.txt").toString().split("\n\n").map(v => new Document({ pageContent: v })));
rag.addDocuments(fs.readFileSync("./llm_lib/data/pmf.txt").toString().split("\n\n").map(v => new Document({ pageContent: v })));
// rag.addDocument(new Document({ pageContent: fs.readFileSync("./llm_lib/data/c_0485.txt").toString() }));
await rag.saveContext();


let server = app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});

const wss = new WebSocketServer({ server });

wss.on("connection", (ws) => {
  let asked = false;

  ws.on("message", async (data) => {
    if (asked) return;
    asked = true;

    await rag.askStream(ws, { question: JSON.parse(data).messages, hour: (new Date()).toLocaleString() });

    ws.close();
  });
});