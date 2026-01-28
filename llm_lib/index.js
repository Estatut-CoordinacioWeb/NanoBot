import { MemoryService }  from './_leagacy_memoryService.js';
import fs from "fs";

const memoryService = new MemoryService("localhost", "gemma3:4b");
await memoryService.storeMemory(fs.readFileSync("./data/organigrama.txt"));
// await memoryService.storeMemoryJSON(fs.readFileSync("./data/c_0485.json"));

// const response = await memoryService.getRelevantMemory("Genera actividades en base a los contenidos del RA1 y RA2 del módulo 0485 Programació. Responde en castellano");
const response = await memoryService.getRelevantMemory("Quién es Javi Gongora?");

// console.log("src:", response.sourceDocuments);
// console.log(response.text);