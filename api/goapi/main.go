package main

import (
	"encoding/json"
	"log"
	"math"
	"net/http"
	"os"
	"strconv"
	"time"
)

type healthResponse struct {
	Service   string    `json:"service"`
	Status    string    `json:"status"`
	Timestamp time.Time `json:"timestamp"`
}

type precisionResponse struct {
	Input  float64 `json:"input"`
	Result float64 `json:"result"`
	Method string  `json:"method"`
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", healthHandler)
	mux.HandleFunc("GET /precision", precisionHandler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	server := &http.Server{
		Addr:              ":" + port,
		Handler:           requestLogger(mux),
		ReadHeaderTimeout: 5 * time.Second,
	}

	log.Printf("Go API listening on :%s", port)
	if err := server.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, healthResponse{
		Service:   "goapi",
		Status:    "ok",
		Timestamp: time.Now().UTC(),
	})
}

func precisionHandler(w http.ResponseWriter, r *http.Request) {
	raw := r.URL.Query().Get("x")
	if raw == "" {
		raw = "1"
	}

	x, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "query parameter x must be a number",
		})
		return
	}

	result := math.Log1p(x) * math.Sqrt(math.Abs(x)+1)
	writeJSON(w, http.StatusOK, precisionResponse{
		Input:  x,
		Result: result,
		Method: "log1p(x) * sqrt(abs(x)+1)",
	})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(value); err != nil {
		log.Printf("write response: %v", err)
	}
}

func requestLogger(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}
