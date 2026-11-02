# Frontend Multi-stage Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json /app/
RUN npm ci

COPY . /app/
RUN npm run build

# Nginx stage to serve static assets
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
# Copy custom nginx.conf if needed, otherwise use default
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
