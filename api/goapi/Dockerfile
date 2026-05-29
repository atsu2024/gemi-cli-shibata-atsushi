FROM golang:1.22-alpine AS builder

WORKDIR /src
COPY go.mod .
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/goapi .

FROM alpine:3.20

RUN adduser -D -H appuser
USER appuser

WORKDIR /app
COPY --from=builder /out/goapi /app/goapi

EXPOSE 8080

CMD ["/app/goapi"]
