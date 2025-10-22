// Package pkg contains miscellaneous single file packages.
package pkg

import (
	"fmt"
	"strings"
	"time"

	"github.com/caarlos0/env/v11"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

type Config struct {
	// for logging
	LogLevel string `env:"LOG_LEVEL" envDefault:"debug"`
	// for Reaper
	ReaperPollEvery  time.Duration `env:"REAPER_POLL_EVERY" envDefault:"10s"`
	ReaperTimeToLive time.Duration `env:"REAPER_TIME_TO_LIVE" envDefault:"30s"`
	// for k8s resouces
	K8SClientset        *kubernetes.Clientset
	K8sNamespace        string `env:"K8S_NAMESPACE" envDefault:"miro-infra"`
	K8sLoadBalancerPort int    `env:"K8S_LOAD_BALANCER_PORT" envDefault:"8080"`
	BaseURLForFaaS      string `env:"BASE_URL_FOR_FAAS" envDefault:"https://faas.miromind.site/faas"` // used for returning the URL to access the function
	// for gateway
	Port               int    `env:"PORT" envDefault:"8080"`
	GatewayServiceName string `env:"GATEWAY_SERVICE_NAME" envDefault:"faas-gateway"`
	GatewayPathPrefix  string `env:"GATEWAY_PATH_PREFIX" envDefault:"/gateway"`
}

// GetConfig parses the environment variables and returns a Config.
func GetConfig() (Config, error) {
	cfg, err := env.ParseAs[Config]()
	if err != nil {
		return Config{}, err
	}

	// hydrate the client set
	config, err := rest.InClusterConfig()
	if err != nil {
		return cfg, fmt.Errorf("rest.InClusterConfig(): %w", err)
	}
	clientSet, err := kubernetes.NewForConfig(config)
	if err != nil {
		return cfg, fmt.Errorf("kubernetes.NewForConfig(): %w", err)
	}
	cfg.K8SClientset = clientSet
	// fmt.Printf("Kubeconfig: %#v\n", config)

	// validate the config
	if !strings.HasPrefix(cfg.GatewayPathPrefix, "/") {
		return cfg, fmt.Errorf("cfg.GatewayPathPrefix MUST start with /")
	}

	if strings.HasSuffix(cfg.GatewayPathPrefix, "/") {
		return cfg, fmt.Errorf("cfg.GatewayPathPrefix MUST NOT end with /")
	}

	if cfg.Port <= 0 {
		return cfg, fmt.Errorf("cfg.Port must be greater than 0")
	}

	return cfg, nil
}
