# Stage 1: Development
FROM node:18-alpine AS development

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

# TypeScript configuration
RUN npm install -D typescript @types/react @types/react-dom @types/node
RUN npm install -D tailwindcss postcss autoprefixer
RUN npx tailwindcss init -p

EXPOSE 3000

CMD ["npm", "run", "dev"]

# Stage 2: Production build
FROM node:18-alpine AS build

WORKDIR /app

COPY package*.json ./

RUN npm install --frozen-lockfile

COPY . .

RUN npm run build

# Stage 3: Production
FROM node:18-alpine AS production

WORKDIR /app

COPY --from=build /app/next.config.js ./
COPY --from=build /app/public ./public
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static

EXPOSE 3000

ENV NODE_ENV production

CMD ["node", "server.js"]
