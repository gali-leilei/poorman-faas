package proxy

import (
	"net/http"
	"net/http/httputil"
	"net/url"
	"path"
	"poorman-faas/pkg/util"
	"strings"
)

// RewriteURL rewrites request URL to lb to request URL to internal service.
//
// Incoming: https://{lb-ip}/api/{svc-name}/{path-suffix}
// Outgoing: http://{svc-name}.{ns}.svc.cluster.local/{path-suffix}
func RewriteURL(pathPrefix string, namespace string, getServiceName func(*http.Request) string) func(*httputil.ProxyRequest) {
	return func(req *httputil.ProxyRequest) {
		serviceName := getServiceName(req.In)
		newPrefix := path.Join("/faas", pathPrefix, serviceName)
		newPath := strings.TrimPrefix(req.In.URL.Path, newPrefix)
		newHost := util.K8SInternalDNSName(namespace, serviceName)

		req.Out.URL = &url.URL{
			Scheme: "http",
			Host:   newHost,
			Path:   newPath,
		}
	}
}
