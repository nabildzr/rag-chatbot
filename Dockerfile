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
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/api ./cmd/api && mkdir -p /out/data

# 3a. Minimal runtime without OCR (~18 MB): docker build --target slim .
FROM gcr.io/distroless/static-debian12:nonroot AS slim
WORKDIR /app
COPY --from=api /out/api /app/api
COPY --from=web /web/dist /app/web
# Uploaded originals; mount a volume here. Owned by the nonroot user (65532).
COPY --from=api --chown=65532:65532 /out/data /app/data
ENV STATIC_DIR=/app/web DATA_DIR=/app/data PORT=8080
EXPOSE 8080
USER nonroot
ENTRYPOINT ["/app/api"]

# 3b. Default runtime with OCR for images and scanned PDFs: Tesseract
# (Indonesian + English) and Poppler for rendering PDF pages.
FROM alpine:3.21 AS full
RUN apk add --no-cache ca-certificates tesseract-ocr tesseract-ocr-data-ind tesseract-ocr-data-eng poppler-utils \
 && adduser -D -H -u 65532 nonroot
WORKDIR /app
COPY --from=api /out/api /app/api
COPY --from=web /web/dist /app/web
COPY --from=api --chown=65532:65532 /out/data /app/data
ENV STATIC_DIR=/app/web DATA_DIR=/app/data PORT=8080
EXPOSE 8080
USER 65532
ENTRYPOINT ["/app/api"]
