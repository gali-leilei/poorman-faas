// Package main is what gets deployed to the cloud platform.
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"poorman-faas/pkg"
	"poorman-faas/pkg/proxy"
	pkg_reaper "poorman-faas/pkg/reaper"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/httplog/v3"
	"github.com/go-chi/httprate"
)

func run(ctx context.Context, cfg pkg.Config, logger *slog.Logger) error {
	// initialize the reaper (which also hydrates from existing cluster resources)
	// for debugging, we set a very short time to live and a very short poll every
	reaper := pkg_reaper.New(ctx, cfg.ReaperPollEvery, cfg.ReaperTimeToLive, cfg.K8SClientset, cfg.K8sNamespace, logger)

	r := chi.NewRouter()
	r.Use(httplog.RequestLogger(logger, nil))
	faas := chi.NewRouter()
	r.Mount("/faas", faas)
	// admin routes: this creates faas service.
	{
		admin := chi.NewRouter()

		// because this creates k8s resource, we are extra careful.
		// for example, see e2b create sandbox rate limit at 5/second.
		admin.Use(httprate.LimitByIP(10, time.Minute))
		admin.Post("/python", getUploadHandler(cfg, reaper,logger))
		faas.Mount("/admin", admin)
	}
	// gateway routes: this proxies to the faas service.
	{
		gateway := chi.NewRouter()
		namespace := cfg.K8sNamespace
		getServiceName := func(r *http.Request) string {
			// return r.PathValue("svcName")
			return chi.URLParam(r, "svcName")
		}
		rp, err := proxy.New(
			proxy.WithTransport(proxy.ProxyTransport()),
			proxy.WithRewrites(
				proxy.RewriteURL(cfg.GatewayPathPrefix, namespace, getServiceName),
				proxy.DebugRequest(logger),
			),
			proxy.WithModifyResponse(func(r *http.Response) error {
				svcName := getServiceName(r.Request)
				reaper.MustUpdate(r.Request.Context(), svcName)
				return nil
			}),
		)
		if err != nil {
			return fmt.Errorf("proxy.New(): %w", err)
		}
		gateway.Handle("/{svcName}/*", rp)
		faas.Mount(cfg.GatewayPathPrefix, gateway)
	}
	// add health check route
	{
		r.Handle("/health", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("ok"))
		}))
	}

	// Single server listening on port 8080
	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", cfg.Port),
		Handler: r,
	}

	// Start the single server
	go func() {
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("Server failed", "error", err)
			return
		}
	}()

	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	// Wait for shutdown signal
	<-ctx.Done()
	logger.Info("Received shutdown signal, shutting down server...")

	// Create a context with a timeout for graceful shutdown
	shutdownCtx := context.Background()
	shutdownCtx, shutdownCancel := context.WithTimeout(shutdownCtx, 10*time.Second)
	defer shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		logger.Error("Error shutting down server", "error", err)
		return err
	}
	return nil
}

func main() {
	config, err := pkg.GetConfig()
	if err != nil {
		panic(err)
	}

	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelDebug}))
	ctx := context.Background()
	err = run(ctx, config, logger)
	if err != nil {
		panic(err)
	}
}
