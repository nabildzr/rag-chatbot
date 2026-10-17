# Production image: Go API serving the built frontend from one origin.

# 1. Frontend
FROM node:24-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# 2. API
FROM golang:1.27-alpine AS api
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cmd/ cmd/
COPY internal/ internal/
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/api ./cmd/api

# 3. Runtime: static binary + CA certificates (for the Groq HTTPS API), non-root
FROM gcr.io/distroless/static-debian12:nonroot
WORKDIR /app
COPY --from=api /out/api /app/api
COPY --from=web /web/dist /app/web
ENV STATIC_DIR=/app/web PORT=8080
EXPOSE 8080
USER nonroot
ENTRYPOINT ["/app/api"]
