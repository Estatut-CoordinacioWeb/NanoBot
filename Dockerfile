FROM node:current-alpine

WORKDIR /estatut/ollama

# Init repository
COPY . .

CMD [ "node", "./server.js" ]
EXPOSE 3039:3039
EXPOSE 11434:11434