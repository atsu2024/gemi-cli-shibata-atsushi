# FastAPI and Go API Docker Deployment

This project includes two standalone API services:

- FastAPI: `http://localhost:8000`
- Go API: `http://localhost:8080`

## Run with Docker Compose

```powershell
docker compose up --build
```

## Verify

```powershell
curl http://localhost:8000/health
curl http://localhost:8080/health
curl -X POST http://localhost:8000/simulate -H "Content-Type: application/json" -d "{\"name\":\"lorenz\",\"steps\":100,\"initial_value\":1.0}"
curl "http://localhost:8080/precision?x=2.5"
```

## Stop

```powershell
docker compose down
```
