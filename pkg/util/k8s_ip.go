package util

import (
	"context"
	"fmt"
	"path"
	"log/slog"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

// K8SInternalDNSName returns the internal domain name (without schema).
//
// {svc-name}.{ns}.svc.cluster.local
func K8SInternalDNSName(namespace string, serviceName string) string {
	// TODO: use url.Parse to check
	return fmt.Sprintf("%s.%s.svc.cluster.local", serviceName, namespace)
}

// K8sExternalDomainName returns the gateway url (with schema).
//
// https://{lb-ip}:{lb-port}/{gateway-prefix}/{svc-name}
func K8sExternalDomainName(ctx context.Context, clientset *kubernetes.Clientset, baseURL string, loadBalancerPort int, gatewayServiceName string, gatewayPrefix string, namespace string, serviceName string) (string, error) {

	if baseURL != "" {
		return baseURL + path.Join(gatewayPrefix, serviceName), nil
	}
	svc, err := clientset.CoreV1().Services(namespace).Get(context.Background(), gatewayServiceName, metav1.GetOptions{})
	if err != nil {
		return "", fmt.Errorf("clientset.CoreV1().Services(%s).Get(%s): %w", namespace, gatewayServiceName, err)
	}

	// TODO: use url.Parse to check
	// TODO: already checked that service is read when creating helm.Chart
	if len(svc.Status.LoadBalancer.Ingress) == 0 {
		return "", fmt.Errorf("svc.Status.LoadBalancer.Ingress is empty")
	}
	LoadBalancerIP := svc.Status.LoadBalancer.Ingress[0].Hostname
	if LoadBalancerIP == "" {
		return "", fmt.Errorf("svc.Status.LoadBalancer.Ingress[0].IP is empty")
	}
	if LoadBalancerIP == "<pending>" {
		return "", fmt.Errorf("svc.Status.LoadBalancer.Ingress[0].IP is pending")
	}
	return fmt.Sprintf("http://%s:%d%s/%s", LoadBalancerIP, loadBalancerPort, gatewayPrefix, serviceName), nil
}

// WaitForServiceHealth waits for the Kubernetes Deployment to become ready by checking
// the deployment status. It waits up to 60 seconds, checking every 5 seconds.
//
// A deployment is considered ready when the number of available replicas equals the desired replicas
// (i.e., ReadyReplicas and AvailableReplicas match the desired count). Returns nil if the deployment becomes ready, or an error if it times out.
func WaitForServiceHealth(ctx context.Context, clientset *kubernetes.Clientset, namespace string, deploymentName string, logger *slog.Logger) error {
	logger.Info("Waiting for deployment to become ready", "deployment", deploymentName, "namespace", namespace)

	deploymentClient := clientset.AppsV1().Deployments(namespace)

	// Wait up to 60 seconds, checking every 5 seconds (12 attempts)
	maxAttempts := 13
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for attempt := 1; attempt <= maxAttempts; attempt++ {
		// Check for context cancellation
		select {
		case <-ctx.Done():
			return fmt.Errorf("context cancelled while waiting for deployment %s: %w", deploymentName, ctx.Err())
		default:
		}

		deployment, err := deploymentClient.Get(ctx, deploymentName, metav1.GetOptions{})
		if err != nil {
			logger.Warn("Failed to get deployment status", "error", err, "attempt", attempt)
			if attempt < maxAttempts {
				select {
				case <-ctx.Done():
					return fmt.Errorf("context cancelled while waiting for deployment %s: %w", deploymentName, ctx.Err())
				case <-ticker.C:
				}
			}
			continue
		}

		// Check if deployment is ready
		// A deployment is ready when:
		// 1. Replicas == ReadyReplicas (all replicas are ready)
		// 2. AvailableReplicas >= Replicas (replicas are available)
		// 3. ReadyReplicas > 0 (at least one replica is ready)
		desiredReplicas := int32(1)
		if deployment.Spec.Replicas != nil {
			desiredReplicas = *deployment.Spec.Replicas
		}

		logger.Info("Checking deployment status",
			"deployment", deploymentName,
			"attempt", attempt,
			"desired", desiredReplicas,
			"ready", deployment.Status.ReadyReplicas,
			"available", deployment.Status.AvailableReplicas,
			"updated", deployment.Status.UpdatedReplicas,
		)

		if deployment.Status.ReadyReplicas >= desiredReplicas &&
			deployment.Status.AvailableReplicas >= desiredReplicas {
			logger.Info("Deployment is ready", "deployment", deploymentName)
			return nil
		}

		if attempt < maxAttempts {
			select {
			case <-ctx.Done():
				return fmt.Errorf("context cancelled while waiting for deployment %s: %w", deploymentName, ctx.Err())
			case <-ticker.C:
			}
		}
	}

	return fmt.Errorf("deployment %s did not become ready within 60 seconds", deploymentName)
}
